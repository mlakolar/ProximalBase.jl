# Known limitations of the numerical contract

*Audited against `master` at `920ebece` ([`920ebece7cea104eba61bb1e9c751c347444c802`](https://github.com/mlakolar/ProximalBase.jl/tree/920ebece7cea104eba61bb1e9c751c347444c802)) on 2026-11-03, with Julia 1.13.3 and 1.10.12 on macOS (Apple M3 Pro); device entries with JLArrays 0.4.2 on Julia 1.13 only. Active status rechecked against `master` at `f9030a64` ([`f9030a649cc0199c8866edd5c317dbc392dd5d5c`](https://github.com/mlakolar/ProximalBase.jl/tree/f9030a649cc0199c8866edd5c317dbc392dd5d5c), after #553 and #549) on 2026-11-04, following the maintainer's review of #558: every case of every active entry prints `PRESENT`, 554 cases on Julia 1.13.3 and the 517 that run on 1.10.12 (JLArrays does not load there); LIM-072 rests on `@test_broken` rows. How the entries were found and checked: [the audit of 2026-11-03](audit-2026-11-03/README.md). To recheck against a later commit, run `julia --project=. docs/design/numerical-contract/audit-2026-11-03/reproduce-active.jl`; it prints one line per behavior (`PRESENT`, `FIXED`, `CHANGED`, `ERROR` or `SKIPPED`) and a summary per entry.*

## What this file is

| source | responsibility |
|:---|:---|
| [`docs/src/numerics.md`](../../src/numerics.md) | the normative contract and its approved exceptions |
| [`decisions.md`](decisions.md) | maintainer decisions; each entry's **Decided** line governs |
| **this file** | a human-reviewed index of behaviors on current `master` that miss a rule, are approved exceptions, or are not verified |
| [`coverage.md`](coverage.md) | the generated conformance matrix; never edited by hand to track limitations |
| GitHub issues, [#366](https://github.com/mlakolar/ProximalBase.jl/issues/366) and its group trackers | the owners of the work |

**Scope.** The entries come from the audit of every closed issue and pull
request (#556), the seed list in #556, and contradictions found while
checking them against `master`. This file does not repeat the #366 checklist
or the conformance matrix, and it holds no fixed bugs: those are listed, with
their evidence, in the audit index
([`audit-2026-11-03/clusters.tsv`](audit-2026-11-03/clusters.tsv)).
Performance limitations (measured or not) are also in the audit index, with
their owners, not here.

**Classes.**

- *Contract gap.* A promise of the contract (or of a docstring) that `master`
  does not keep, reproduced at the SHA above. Tracking is not acceptance: an
  entry stays a gap until it is fixed or a maintainer decision approves it.
- *Approved limitation.* Permitted by a specific maintainer decision and
  stated in the user documentation. The decision is linked.
- *Coverage unverified.* Not established either way, for the stated reason.
  Not a confirmed defect and not a guarantee.
- *Housekeeping.* A confirmed defect that no rule of the contract covers,
  such as the text of a diagnostic.

Behaviors the contract excludes on purpose are listed separately under
[Intentional exclusions](#intentional-exclusions); they are not defects.

## Maintenance rule

- A pull request that **discovers, fixes, explicitly defers or accepts** a
  limitation updates this file in the same pull request: add, edit, move or
  remove the entry, and keep `reproduce-active.jl` in step. IDs are stable and
  never reused. An entry split after publication keeps its number with a
  letter suffix (LIM-048a, LIM-048b, LIM-048c), so that a reference to the
  number still finds it.
- `reproduce-active.jl` has one case per behavior, tagged with the entry and
  the behavior of the audit index it comes from. A case prints `PRESENT` when
  the limitation shows exactly as described, `FIXED` when the behavior the
  contract requires shows instead, and `CHANGED` when neither does.
- A **fix** removes the entry from the active sections and adds a line to
  [Resolved since the audit](#resolved-since-the-audit) with the pull request
  and its evidence: every case of the entry prints `FIXED`, and a regression
  test pins the new behavior. A fix of some of the cases narrows the entry.
- **Accepting** a limitation needs a maintainer decision recorded in
  `decisions.md` (a **Decided** line) and the user documentation; then the
  entry moves to [Approved limitations](#approved-limitations) with the link.
  Changing a contract promise is a separate decision; this file never changes
  normative text. Writing a proposed release impact or a deferral here
  approves nothing.
- Closing an issue or merging a pull request does not change an entry by
  itself. Each entry names one owning issue; related issues and pull requests
  are linked, not copied.

## Decisions

These questions came out of the audit. Each is written so that it can be
answered on [#556](https://github.com/mlakolar/ProximalBase.jl/issues/556) or
the linked issue; this file changes only after the answer. Q1 is decided; Q2
and Q3 are open, and until they are answered every release impact below is a
proposal, not an approved deferral.

- **Q1. Block sums and failures found while computing (LIM-001, LIM-003,
  LIM-005, LIM-006). Decided** in the
  [maintainer's review of #558](https://github.com/mlakolar/ProximalBase.jl/pull/558#pullrequestreview-5769915868)
  (2026-11-04): decision 42 stands, and the current behavior is an
  implementation gap, not an accepted exception. Pull request #543 had added
  a paragraph to O6 in `numerics.md` saying that in `SeparableSum`,
  `SliceSum` and `FiberSum` a later block's failure of kind 2, 3 or 4 "leaves
  the earlier blocks written". No **Decided** line approved it, and it
  contradicted decision 42 ("Infeasibility found during a search raises
  `ArgumentError` before writing. An arithmetic stopgap raises the dedicated
  exception before writing.") and O6.1–O6.3. #558 replaces it: the normative
  paragraph now says that before writing means before the first block is
  written, for failures of kinds 1 to 3 (kind 4 keeps O6.4's no-rollback
  rule), and an implementation-status note lists the failures that still
  leave earlier blocks written and points here. The `SeparableSum` docstring
  (`src/calculus.jl`, "It does not cover every failure") still describes the
  current behavior; the pull requests that fix these entries update it. The
  audit also found that `Conjugate`'s finite-result failure (LIM-001) is not
  confined to invalid or endpoint steps: it fires at valid positive steps such
  as `γ = 1e-10` with `x = [1e300]`. LIM-002, LIM-005 and LIM-006 are
  preflightable; LIM-001 disappears at valid steps once LIM-069 (#255) is
  fixed, leaving only invalid steps (LIM-002); only LIM-003 may need a bounded
  staged commit, whose cost can be measured on that one family.
- **Q2 (open). Examples and 0.6 promises on the contract page that `master` does not
  meet (LIM-016, LIM-023, LIM-038, LIM-039, LIM-045, LIM-047, LIM-056,
  LIM-082).** The page's status note says parts are still being implemented,
  but these examples are written as facts: V10 (LIM-039), T3's
  `QuadraticFunction` gradient (LIM-038), R1's `Translate` example (LIM-023),
  P3's `[1e-50, 2e-50]` metric (LIM-016), E2's `NormL2` example (LIM-047), T6's
  "in 0.6, `Float16` is rejected" (LIM-045), and O3.2's "until 0.6 is
  complete" (LIM-082). E2 already marks its `γ = Inf` example (LIM-056) as not
  yet conforming. Decide whether 0.6.0 requires these, or whether the page
  marks each unmet example the same way. *Recommendation (not a decision):*
  block the release on the silent-wrong-result entries (LIM-013, LIM-016,
  LIM-047, LIM-048a, LIM-048b, LIM-048c), and mark the other unmet examples on
  the page in the same way as LIM-056 until their groups land. Until Q2 is
  answered, none of these is approved as a deferral, and the page's examples
  stay unmet promises.
- **Q3 (open). The 0.6 iterate redesign (LIM-071, LIM-074, LIM-076, LIM-077).**
  These are approved 0.6 design items (FD-9, OWN-1/OWN-3/ACT-2, FW-2, FC-3)
  that are not implemented. Decide whether they are in 0.6.0 (#456, #391).
  Until then their release impact is open.
- **Housekeeping (no contract change).** #216: its original reproducer is
  fixed (#245) and its block-step preflights are done (#543); what remains is
  LIM-001 and LIM-005, which it owns. #239 is open, but its last status says
  that what remains is #274, which is closed; LIM-005 is owned by #216, so
  #239 can be closed or re-scoped. #307 is open, but its reproduction passes
  on `master` (see the audit index); check and close. Tests that pin
  behavior the contract has since changed (`test/spectral_nonfinite_values.jl:36`,
  `test/infinite_weights.jl:58`, `test/threshold_factor_type.jl:144`,
  `test/separable.jl:413`) must change with LIM-049, LIM-052, LIM-048c and
  LIM-101; the three `VERSION < v"1.11"` allocation skips in
  `test/conjugate_projections.jl` (lines 562, 577, 627) no longer allocate on
  Julia 1.10.12. LIM-079, first listed as a gap, is a diagnostic message with
  no contract rule; it is kept below as a
  [housekeeping item](#lim-079), with its check in `reproduce-active.jl`.

## Overview

Priority: P1 = silently wrong result or false feasibility that a caller cannot detect; P2 = contract gap on exceptional or extreme inputs, a wrong exception class, or a partial write; P3 = narrow, cosmetic, or test helpers. The 0.6 column is a proposal pending the maintainer's decision.

| ID | limitation | rule | class | owner | priority | 0.6 (proposed) |
|:---|:---|:---|:---|:---|:---|:---|
| [LIM-001](#lim-001) | Conjugate finite-result check at valid steps: error instead of result; earlier blocks written | O6.3, decision 42 | gap | [#216](https://github.com/mlakolar/ProximalBase.jl/issues/216) | P2 | deferral |
| [LIM-002](#lim-002) | Conjugate block at invalid or unnamed endpoint step caught after earlier blocks written | P1, O6.1 | gap | [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311) | P2 | deferral |
| [LIM-003](#lim-003) | LinearConstrained search-found infeasibility in a later block writes earlier blocks | O6.2, decision 42 | gap | [#323](https://github.com/mlakolar/ProximalBase.jl/issues/323) | P2 | deferral |
| [LIM-004](#lim-004) | Empty LinearConstrained intersection returns NaN, no error, at a NaN input or γ = Inf | O6.2, E1, decision 42 | gap | [#323](https://github.com/mlakolar/ProximalBase.jl/issues/323) | P2 | deferral |
| [LIM-005](#lim-005) | Non-proximable later block raises MethodError after earlier blocks written | O6.1 | gap | [#216](https://github.com/mlakolar/ProximalBase.jl/issues/216) | P3 | deferral |
| [LIM-006](#lim-006) | Tuple Conjugate destination conversion fails after earlier blocks written | O6.1, T1 | gap | [#371](https://github.com/mlakolar/ProximalBase.jl/issues/371) | P2 | deferral |
| [LIM-007](#lim-007) | quadratic_model! nested block mismatch updates earlier siblings first | E1, O6.1 | gap | [#495](https://github.com/mlakolar/ProximalBase.jl/issues/495) | P3 | deferral |
| [LIM-008](#lim-008) | Dense Symmetric destination fails inside Base after a partial write | O6.1, T1 | gap | [#495](https://github.com/mlakolar/ProximalBase.jl/issues/495) | P3 | deferral |
| [LIM-009](#lim-009) | Integer destinations: InexactError, sparse route writes first | T1, O6.1, decision 8 | gap | [#492](https://github.com/mlakolar/ProximalBase.jl/issues/492) | P2 | deferral |
| [LIM-010](#lim-010) | Complex sources into iterates and their views bypass the real-only check | S1, O6.1 | gap | [#364](https://github.com/mlakolar/ProximalBase.jl/issues/364) | P3 | deferral |
| [LIM-011](#lim-011) | Complex steps and scalars raise MethodError, not ArgumentError | S1 | gap | [#365](https://github.com/mlakolar/ProximalBase.jl/issues/365) | P3 | deferral |
| [LIM-012](#lim-012) | Complex adjoint product with a SparseIterate returns a result | S1 | gap | [#492](https://github.com/mlakolar/ProximalBase.jl/issues/492) | P3 | deferral |
| [LIM-013](#lim-013) | Negative and NaN steps accepted by most leaves and entry routes | P1, P2, O6.1, decision 25 | gap | [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311) | P1 | blocks 0.6 |
| [LIM-014](#lim-014) | Value-dependent or wrong-class step rejections (DomainError) | P1 | gap | [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311) | P2 | deferral |
| [LIM-015](#lim-015) | Unnamed endpoint steps give NaN instead of a limit or rejection | P1, E4, decision 26 | gap | [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311) | P2 | deferral |
| [LIM-016](#lim-016) | Subnormal and non-representable diagonal steps give infeasible or -Inf results | P3, P1, R1, decision 27 | gap | [#257](https://github.com/mlakolar/ProximalBase.jl/issues/257) | P1 | blocks 0.6 |
| [LIM-017](#lim-017) | Multiplier searches are not certified; PrecisionError never raised | A3, A1, O6.3 | gap | [#151](https://github.com/mlakolar/ProximalBase.jl/issues/151) | P2 | deferral |
| [LIM-018](#lim-018) | Search reports 'misses the domain' for feasible problems | A3, O6.2 | gap | [#475](https://github.com/mlakolar/ProximalBase.jl/issues/475) | P2 | deferral |
| [LIM-019](#lim-019) | Ridge-plus-hyperplane search: spurious 'misses the domain' at large γλ | A3, O6.2 | gap | [#510](https://github.com/mlakolar/ProximalBase.jl/issues/510) | P2 | deferral |
| [LIM-020](#lim-020) | Hyperplane/halfspace coefficient rounded to a narrow output | T3, T4, R2 | gap | [#218](https://github.com/mlakolar/ProximalBase.jl/issues/218) | P2 | deferral |
| [LIM-021](#lim-021) | Hyperplane/halfspace projection returns -Inf when ⟨a, x⟩ overflows | R2, R4 | gap | [#563](https://github.com/mlakolar/ProximalBase.jl/issues/563) | P2 | deferral |
| [LIM-022](#lim-022) | Scalar kernels and losses lose range before the weight | R1 | gap | [#356](https://github.com/mlakolar/ProximalBase.jl/issues/356) | P2 | deferral |
| [LIM-023](#lim-023) | Translate/Precompose of norms do not recover through homogeneity | R1, decision 45 | gap | [#357](https://github.com/mlakolar/ProximalBase.jl/issues/357) | P2 | decide Q2 |
| [LIM-024](#lim-024) | Values of sums and calculus wrappers lose range | R1, decision 45 | gap | [#359](https://github.com/mlakolar/ProximalBase.jl/issues/359) | P2 | deferral |
| [LIM-025](#lim-025) | value_and_prox! of Scaled returns 0 when the inner value underflows | O2, R1 | gap | [#359](https://github.com/mlakolar/ProximalBase.jl/issues/359) | P2 | deferral |
| [LIM-026](#lim-026) | Norms rounded before a large weight; signed reductions overflow | R1 | gap | [#360](https://github.com/mlakolar/ProximalBase.jl/issues/360) | P2 | deferral |
| [LIM-027](#lim-027) | Proximal outputs overflow at representable results (Berhu, HingeLoss, NormL1Fused) | R2 | gap | [#361](https://github.com/mlakolar/ProximalBase.jl/issues/361) | P2 | deferral |
| [LIM-028](#lim-028) | Exponential-cone projections wrong when an output component overflows | R4, R2 | gap | [#361](https://github.com/mlakolar/ProximalBase.jl/issues/361) | P3 | deferral |
| [LIM-029](#lim-029) | Weighted NormL1 value loses underflowing products | R1 | gap | [#373](https://github.com/mlakolar/ProximalBase.jl/issues/373) | P2 | deferral |
| [LIM-030](#lim-030) | Spectral values overflow at an overflowing spectrum | R1, decision 45 | gap | [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283) | P2 | deferral |
| [LIM-031](#lim-031) | Quadratic values lose accuracy when (Ax)ᵢ underflows | R1 | gap | [#520](https://github.com/mlakolar/ProximalBase.jl/issues/520) | P3 | deferral |
| [LIM-032](#lim-032) | GLMLoss values do not recover out-of-range weights and predictors | R1 | gap | [#358](https://github.com/mlakolar/ProximalBase.jl/issues/358) | P2 | deferral |
| [LIM-033](#lim-033) | GroupMCP/GroupSCAD prox NaN when the group norm overflows | R2 | gap | [#534](https://github.com/mlakolar/ProximalBase.jl/issues/534) | P2 | deferral |
| [LIM-035](#lim-035) | Poisson quadratic model: Inf weight and NaN response at η ≥ 710 | S2, O6.3, decision 43 | gap | [#317](https://github.com/mlakolar/ProximalBase.jl/issues/317) | P2 | deferral |
| [LIM-036](#lim-036) | Constrained(NormL1, IndSphereL2) throws when γλw overflows | R3 | gap | [#484](https://github.com/mlakolar/ProximalBase.jl/issues/484) | P2 | deferral |
| [LIM-037](#lim-037) | Smoothed deriv2 is a finite difference with undocumented accuracy | A1 | gap | [#491](https://github.com/mlakolar/ProximalBase.jl/issues/491) | P3 | deferral |
| [LIM-038](#lim-038) | Narrow data at a wide argument computed at data precision | T3, T4, O1, decision 6 | gap | [#487](https://github.com/mlakolar/ProximalBase.jl/issues/487) | P2 | decide Q2 |
| [LIM-039](#lim-039) | V10: narrow parameters combined in the narrow type for a wider destination | T4, T3 | gap | [#554](https://github.com/mlakolar/ProximalBase.jl/issues/554) | P2 | decide Q2 |
| [LIM-040](#lim-040) | Threshold and coefficient products formed in the factors' type | T3 | gap | [#519](https://github.com/mlakolar/ProximalBase.jl/issues/519) | P2 | deferral |
| [LIM-041](#lim-041) | AtomIterate coordinate and block steps round a wider argument | T3 | gap | [#513](https://github.com/mlakolar/ProximalBase.jl/issues/513) | P3 | deferral |
| [LIM-042](#lim-042) | Group value_and_prox! is not f(y) for a narrower argument and wider destination | O2, T3 | gap | [#533](https://github.com/mlakolar/ProximalBase.jl/issues/533) | P3 | deferral |
| [LIM-043](#lim-043) | Group norms computed in the narrower argument type | T3 | gap | [#484](https://github.com/mlakolar/ProximalBase.jl/issues/484) | P3 | deferral |
| [LIM-044](#lim-044) | Precompose around a value-less inner function returns the value in the image precision | O2, T3 | gap | [#490](https://github.com/mlakolar/ProximalBase.jl/issues/490) | P2 | deferral |
| [LIM-045](#lim-045) | Float16 not rejected by role | T6, S1, decision 7 | gap | [#565](https://github.com/mlakolar/ProximalBase.jl/issues/565) | P2 | decide Q2 |
| [LIM-046](#lim-046) | LAPACK functions accept Float16, BigFloat and integer arguments | A5, T6, T7 | gap | [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489) | P2 | deferral |
| [LIM-047](#lim-047) | NormL2/GroupNormL2 prox turn NaN into zeros | E2, decision 20 | gap | [#559](https://github.com/mlakolar/ProximalBase.jl/issues/559) | P1 | blocks 0.6 |
| [LIM-048a](#lim-048a) | NormL0/IndBallL0 select a finite branch or declare feasibility at NaN | E2, decision 20 | gap | [#560](https://github.com/mlakolar/ProximalBase.jl/issues/560) | P1 | blocks 0.6 |
| [LIM-048b](#lim-048b) | LogPenalty/CheckLoss/HingeLoss prox and MCP/SCAD values finite at NaN | E2, decision 20 | gap | [#561](https://github.com/mlakolar/ProximalBase.jl/issues/561) | P1 | blocks 0.6 |
| [LIM-048c](#lim-048c) | SLOPE/NormL1Fused/IndIsotonic outputs finite where they depend on a NaN | E2, decision 20 | gap | [#562](https://github.com/mlakolar/ProximalBase.jl/issues/562) | P1 | blocks 0.6 |
| [LIM-049](#lim-049) | Standalone values at a NaN argument are Inf or ignore the NaN | E2, decision 20 | gap | [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) | P2 | deferral |
| [LIM-050](#lim-050) | value_and_prox! of indicators returns 0 at a stored NaN output | O2, E2 | gap | [#349](https://github.com/mlakolar/ProximalBase.jl/issues/349) | P2 | deferral |
| [LIM-051](#lim-051) | Precompose around an indicator loses the Inf/NaN class | O2, E5 | gap | [#350](https://github.com/mlakolar/ProximalBase.jl/issues/350) | P2 | deferral |
| [LIM-052](#lim-052) | Infinite-weight fixed coordinate is NaN at a NaN or infinite argument | E2, decision 48 | gap | [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) | P2 | deferral |
| [LIM-053](#lim-053) | Zero level is not the zero function at non-finite arguments | E4, decision 22 | gap | [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) | P2 | deferral |
| [LIM-054](#lim-054) | MoreauEnvelope value is NaN at an infinite entry | E3 | gap | [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) | P3 | deferral |
| [LIM-055](#lim-055) | Decision 48 limits not implemented for the remaining families | E3, decision 48 | gap | [#469](https://github.com/mlakolar/ProximalBase.jl/issues/469) | P2 | deferral |
| [LIM-056](#lim-056) | γ = Inf endpoint with a NaN argument returns NaN for a constant map | E2, decision 48 | gap | [#469](https://github.com/mlakolar/ProximalBase.jl/issues/469) | P2 | deferral |
| [LIM-057](#lim-057) | Scaled(indicator, Inf) value is NaN | O2, E4 | gap | [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311) | P2 | deferral |
| [LIM-058](#lim-058) | Scaled(f, 0) rejected for total-domain functions | E4, S4 | gap | [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) | P3 | deferral |
| [LIM-059](#lim-059) | Fused spectral values disagree with the stored output | O2, A5 | gap | [#564](https://github.com/mlakolar/ProximalBase.jl/issues/564) | P2 | deferral |
| [LIM-060](#lim-060) | Device ball/simplex kernels do not classify NaN/Inf and accept bad steps | G1, E2, E3, P1 | gap | [#494](https://github.com/mlakolar/ProximalBase.jl/issues/494) | P2 | deferral |
| [LIM-061](#lim-061) | Device Conjugate(NormLinf/GroupNormLinf) cancels | G1, R2 | gap | [#372](https://github.com/mlakolar/ProximalBase.jl/issues/372) | P2 | deferral |
| [LIM-062](#lim-062) | Device kernels narrow a host level before the weight | G5, G1, R3 | gap | [#322](https://github.com/mlakolar/ProximalBase.jl/issues/322) | P2 | deferral |
| [LIM-063](#lim-063) | Wrappers index scalars on device arrays | G1, S4 | gap | [#327](https://github.com/mlakolar/ProximalBase.jl/issues/327) | P3 | deferral |
| [LIM-064](#lim-064) | Constrained(f, IndBox) value_and_prox! throws for an inner function without a value | O2 | gap | [#316](https://github.com/mlakolar/ProximalBase.jl/issues/316) | P2 | deferral |
| [LIM-065](#lim-065) | Parameter-array shape rules differ across wrappers | E1 | gap | [#312](https://github.com/mlakolar/ProximalBase.jl/issues/312) | P3 | deferral |
| [LIM-066](#lim-066) | restrict(IndBinary) requires every bound pair to contain zero | O5 | gap | [#264](https://github.com/mlakolar/ProximalBase.jl/issues/264) | P3 | deferral |
| [LIM-067](#lim-067) | Conjugate(SingularValue) into a SymmetricSparseIterate throws | A5, M2 | gap | [#407](https://github.com/mlakolar/ProximalBase.jl/issues/407) | P2 | deferral |
| [LIM-068](#lim-068) | Fixed default tolerances independent of precision | A4 | gap | [#396](https://github.com/mlakolar/ProximalBase.jl/issues/396) | P3 | deferral |
| [LIM-069](#lim-069) | Conjugate on the Moreau route throws or misreports at extreme valid steps | R1, R2, R3, P1, O6.3 | gap | [#255](https://github.com/mlakolar/ProximalBase.jl/issues/255) | P2 | deferral |
| [LIM-070](#lim-070) | Conjugate on the Moreau route loses results to cancellation | R2, A1 | gap | [#255](https://github.com/mlakolar/ProximalBase.jl/issues/255) | P2 | deferral |
| [LIM-071](#lim-071) | Strong zeros (FD-9) not implemented in products | M1 | gap | [#428](https://github.com/mlakolar/ProximalBase.jl/issues/428) | P2 | decide Q3 |
| [LIM-072](#lim-072) | Sparse value oracles read every entry | M1 | gap | [#341](https://github.com/mlakolar/ProximalBase.jl/issues/341) | P3 | deferral |
| [LIM-073](#lim-073) | SymmetricSparseIterate scaling by Inf/NaN/0 depends on storage | M1, FN-9 | gap | [#391](https://github.com/mlakolar/ProximalBase.jl/issues/391) | P2 | deferral |
| [LIM-074](#lim-074) | SymmetricSparseIterate is still an AbstractMatrix for generic position loops | M2, OWN-1, OWN-3, ACT-2 | gap | [#391](https://github.com/mlakolar/ProximalBase.jl/issues/391) | P2 | decide Q3 |
| [LIM-075](#lim-075) | copyto! into views and reshapes of a SymmetricSparseIterate is unchecked | M2 | gap | [#368](https://github.com/mlakolar/ProximalBase.jl/issues/368) | P2 | deferral |
| [LIM-076](#lim-076) | fill!(x, 0) and x .= 0 empty the working set | FW-2, FW-3 | gap | [#424](https://github.com/mlakolar/ProximalBase.jl/issues/424) | P2 | decide Q3 |
| [LIM-077](#lim-077) | coordinates(::SparseIterate) returns x itself | FC-3, FD-5 | gap | [#425](https://github.com/mlakolar/ProximalBase.jl/issues/425) | P3 | decide Q3 |
| [LIM-078](#lim-078) | AtomIterate group vectors are shared mutable state | FC-6, FD-6, FD-11 | gap | [#426](https://github.com/mlakolar/ProximalBase.jl/issues/426) | P3 | deferral |
| [LIM-079](#lim-079) | Asymmetry diagnostic prints NaN for BigFloat beyond the Float64 range | none | housekeeping | [#391](https://github.com/mlakolar/ProximalBase.jl/issues/391) | P3 | — |
| [LIM-080](#lim-080) | Testing.test_gradient does not check leaf element types | T1 | gap | [#159](https://github.com/mlakolar/ProximalBase.jl/issues/159) | P3 | deferral |
| [LIM-081](#lim-081) | IndIntersection is approximate; its value 0 is not a feasibility certificate | O3.1 | approved | — | — | — |
| [LIM-082](#lim-082) | LogDetLoss fused value can be finite at a stored output that rounds to singular | O3.2 | approved | — | — | decide Q2 |
| [LIM-083](#lim-083) | R1 does not cover wrappers whose transformed argument overflows (non-norms) | R1 (wrappers) | approved | — | — | — |
| [LIM-084](#lim-084) | Translate/Tilt round between stages in mixed precision | O4 | approved | — | — | — |
| [LIM-085](#lim-085) | Matrix reconstructions may saturate within their normwise bound | R4 (last sentence) | approved | — | — | — |
| [LIM-086](#lim-086) | Summation order of iterate products unspecified; finite overflow order-dependent | A6, decision 15 | approved | — | — | — |
| [LIM-087](#lim-087) | Zero scaling rejected where no domain-preserving representation exists | E4 | approved | — | — | — |
| [LIM-088](#lim-088) | Real GPU hardware not tested (JLArrays only) | — | unverified | [#494](https://github.com/mlakolar/ProximalBase.jl/issues/494) | — | — |
| [LIM-089](#lim-089) | Adapt rules possibly missing for functions with array parameters | — | unverified | [#494](https://github.com/mlakolar/ProximalBase.jl/issues/494) | — | — |
| [LIM-090](#lim-090) | Partial state after a user array throws in coordinate updates | — | unverified | [#495](https://github.com/mlakolar/ProximalBase.jl/issues/495) | — | — |
| [LIM-091](#lim-091) | Parameter or step arrays that are the output (O7 silent) | — | unverified | [#367](https://github.com/mlakolar/ProximalBase.jl/issues/367) | — | — |
| [LIM-092](#lim-092) | Inexact writes into integer iterates from dense sources | — | unverified | [#495](https://github.com/mlakolar/ProximalBase.jl/issues/495) | — | — |
| [LIM-093](#lim-093) | Matrix products overflow before alpha < 1 rescales them | — | unverified | [#366](https://github.com/mlakolar/ProximalBase.jl/issues/366) | — | — |
| [LIM-094](#lim-094) | Data-loss gradients overflow in matrix products | — | unverified | [#542](https://github.com/mlakolar/ProximalBase.jl/issues/542) | — | — |
| [LIM-095](#lim-095) | Conjugate support fallback evaluates user oracles in eltype(y) | — | unverified | [#187](https://github.com/mlakolar/ProximalBase.jl/issues/187) | — | — |
| [LIM-096](#lim-096) | Spectral prox outputs at an overflowing spectrum, f without a scaling rule | — | unverified | [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283) | — | — |
| [LIM-097](#lim-097) | IndGraph construction at large-scale data | — | unverified | [#487](https://github.com/mlakolar/ProximalBase.jl/issues/487) | — | — |
| [LIM-098](#lim-098) | IndAffine rank truncation tolerance | — | unverified | [#493](https://github.com/mlakolar/ProximalBase.jl/issues/493) | — | — |
| [LIM-099](#lim-099) | LogDetLoss second-order oracles disagree off the domain | — | unverified | [#491](https://github.com/mlakolar/ProximalBase.jl/issues/491) | — | — |
| [LIM-100](#lim-100) | Infinite level with a zero step: NormL1 and ElasticNet differ | — | unverified | [#486](https://github.com/mlakolar/ProximalBase.jl/issues/486) | — | — |
| [LIM-101](#lim-101) | An infinite entry is a member of an unbounded box | — | unverified | [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) | — | — |
| [LIM-102](#lim-102) | mul! into a SymmetricSparseIterate with alpha = Inf, empty inner dimension | — | unverified | [#391](https://github.com/mlakolar/ProximalBase.jl/issues/391) | — | — |
| [LIM-103](#lim-103) | Class of Symmetric * X * Symmetric for a sparse iterate (FD-16) | — | unverified | [#419](https://github.com/mlakolar/ProximalBase.jl/issues/419) | — | — |
| [LIM-104](#lim-104) | NormL0 near-tie decisions on the rounded square | — | unverified | [#519](https://github.com/mlakolar/ProximalBase.jl/issues/519) | — | — |
| [LIM-105](#lim-105) | FiberSum(symmetric = true) on an asymmetric input | — | unverified | [#342](https://github.com/mlakolar/ProximalBase.jl/issues/342) | — | — |
| [LIM-106](#lim-106) | Concurrent value calls on one AtomIterate | — | unverified | [#456](https://github.com/mlakolar/ProximalBase.jl/issues/456) | — | — |
| [LIM-107](#lim-107) | Spectral wrappers of weighted penalties not audited for T3/R2 | — | unverified | [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489) | — | — |
| [LIM-108](#lim-108) | Testing.test_separable skips value checks silently | — | unverified | [#524](https://github.com/mlakolar/ProximalBase.jl/issues/524) | — | — |

## Contract gaps

### Failures and partial writes (O6)

#### LIM-001

- **Summary:** A later `Conjugate` block whose finite-result check fails at a valid positive step raises after the earlier blocks were written.
- **Affected API:** `prox!`, `value_and_prox!` of `SeparableSum`, `SliceSum`, `FiberSum` with a `Conjugate` block on the Moreau route
- **Conditions:** valid finite positive step at which `x/γ` overflows or a step-free construction is missing, e.g. `Conjugate(SqrNormL2())` at `[1e300]`, `γ = 1e-10`, or at `γ = 1e-310`
- **Expected:** O6.3 and decision 42: the precision failure is raised as `PrecisionError` before any block is written (better, LIM-069 removes the failure)
- **Actual:** `ArgumentError` ("the proximal point of the conjugate is not finite at a finite input") after block 1 was overwritten; standalone the output is unchanged
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: throws `ArgumentError`; `y[1]` changed; same in a `SliceSum` slice. At `f9030a64`: 4 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  s = SeparableSum(SqrNormL2(2.0), Conjugate(SqrNormL2()))
  y = ([7.0], [7.0])
  prox!(y, s, ([3.0], [1e300]), 1e-10)   # throws; y[1] is now [2.9999999994] = 3/(1 + 2e-10)
  ```
- **Classification:** contract gap (O6.3, decision 42; Q1 decided: a gap, not an exception)
- **Tracking:** owner [#216](https://github.com/mlakolar/ProximalBase.jl/issues/216); related [#255](https://github.com/mlakolar/ProximalBase.jl/issues/255), [#543](https://github.com/mlakolar/ProximalBase.jl/issues/543), [#495](https://github.com/mlakolar/ProximalBase.jl/issues/495)
- **Priority:** P2 — partial write on valid input at extreme scale; the failure is loud
- **Release impact:** Proposed: candidate for deferral (Q1 settled that this is a gap; the release impact is not decided)
- **Resolution criteria:** a later `Conjugate` block that cannot represent its result leaves every earlier block unchanged (and raises `PrecisionError`), or returns the finite result

#### LIM-002

- **Summary:** A later `Conjugate` block at an invalid or unnamed endpoint step is rejected only after the earlier blocks were written.
- **Affected API:** block sums with `Conjugate(f)` on the Moreau route (e.g. `Conjugate(SqrNormL2())`, `Conjugate(IndBallL1(1))` at `γ = 0` or `Inf`)
- **Conditions:** `γ` negative, NaN, `0` or `Inf` where the conjugate names no endpoint
- **Expected:** P1 and O6.1: rejected before any block is written (a preflightable step check)
- **Actual:** `Conjugate(SqrNormL2())` blocks write block 1 for every one of `-1`, `NaN`, `0`, `Inf`; `Conjugate(IndBallL1)` blocks reject `NaN` and `-1` first but write and then raise at `0` and `Inf` for an argument with a zero entry such as `[0, 1]` (at `[1.0]`, `γ = 0` returns the identity without error); `Conjugate(HingeLoss())` accepts `γ = -1`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: throws after block 1 was written. At `f9030a64`: 7 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  s = SeparableSum(SqrNormL2(2.0), Conjugate(SqrNormL2()))
  prox!(([7.0], [7.0]), s, ([3.0], [1.0]), -1.0)
  ```
- **Classification:** contract gap (P1, O6.1)
- **Tracking:** owner [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311); related [#216](https://github.com/mlakolar/ProximalBase.jl/issues/216), [#543](https://github.com/mlakolar/ProximalBase.jl/issues/543)
- **Priority:** P2 — invalid steps escape the block preflight
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** block `prox!` of `Conjugate(SqrNormL2())` at `γ ∈ {-1, NaN, 0, Inf}` throws before writing block 1

#### LIM-003

- **Summary:** Infeasibility that a `LinearConstrained` search discovers in a later block is raised after the earlier blocks were written.
- **Affected API:** block sums with `Constrained(f, IndHyperplane/IndHalfspace/IndHyperslab)`
- **Conditions:** the O(1) pre-check abstains: array normal (`IndHyperplane(ones(2), 5)` with `IndBox(0, 1)`), `BoxConstrained` inner function, overflowing box (`IndBox(-1e308, 1)`)
- **Expected:** O6.2 and decision 42: `ArgumentError` naming the infeasibility before any block is written
- **Actual:** standalone: the error, output unchanged; in a block: the error after block 1 was written
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: throws "the IndHyperplane misses the domain of IndBox"; block 1 written. At `f9030a64`: 5 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  s = SeparableSum(SqrNormL2(2.0), Constrained(IndBox(0.0, 1.0), IndHyperplane(ones(2), 5.0)))
  prox!(([7.0], [7.0, 7.0]), s, ([3.0], [0.5, 0.5]), 1.0)
  ```
- **Classification:** contract gap (O6.2, decision 42; Q1 decided: a gap, not an exception)
- **Tracking:** owner [#323](https://github.com/mlakolar/ProximalBase.jl/issues/323); related [#216](https://github.com/mlakolar/ProximalBase.jl/issues/216), [#304](https://github.com/mlakolar/ProximalBase.jl/issues/304), [#543](https://github.com/mlakolar/ProximalBase.jl/issues/543)
- **Priority:** P2 — partial write when the constraint set is empty
- **Release impact:** Proposed: candidate for deferral (Q1 settled that this is a gap; the release impact is not decided)
- **Resolution criteria:** the block call leaves block 1 unchanged for the three cases of #323 and the array-normal case

#### LIM-004

- **Summary:** An empty `LinearConstrained` intersection returns NaN, with no error, at a NaN argument entry or at `γ = Inf`.
- **Affected API:** `prox!`, `value_and_prox!` of `Constrained(IndBox, IndHyperplane)`
- **Conditions:** set empty for every argument (`IndBox(0, 1)` with `x₁ + x₂ = 5`); argument with a NaN entry, or the indicator endpoint `γ = Inf`
- **Expected:** O6.2 / decision 42: `ArgumentError` naming the infeasibility; E1 orders constraining parameters before non-finite argument entries
- **Actual:** `[NaN, NaN]` with no error, and `value_and_prox!` returns the value `0.0` with that output (O2); at `γ = 1` the same set raises the infeasibility error
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[NaN, NaN]` twice. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  f = Constrained(IndBox(0.0, 1.0), IndHyperplane(1.0, 5.0))
  prox(f, [NaN, 0.0], 1.0); prox(f, [0.5, 0.5], Inf)
  ```
- **Classification:** contract gap (O6.2, E1, decision 42)
- **Tracking:** owner [#323](https://github.com/mlakolar/ProximalBase.jl/issues/323); related [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311), [#216](https://github.com/mlakolar/ProximalBase.jl/issues/216)
- **Priority:** P2 — an infeasible problem is reported as a NaN result
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** both calls raise the infeasibility `ArgumentError` before writing

#### LIM-005

- **Summary:** A non-proximable later block raises `MethodError` after the earlier blocks were written.
- **Affected API:** `prox!`, `value_and_prox!` of block sums with a block that has no proximal operator (e.g. `LogisticLoss`)
- **Conditions:** any step; `isproximable` of the sum is already `false`
- **Expected:** O6.1 (types): an unsupported call fails before any block is written
- **Actual:** `MethodError` from the block's `prox!` after block 1 was overwritten
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `MethodError`; block 1 written. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  s = SeparableSum(SqrNormL2(2.0), LogisticLoss([1.0, 0.0], [1.0 0.0; 0.0 1.0]))
  prox!(([7.0], [7.0, 7.0]), s, ([3.0], [0.1, 0.2]), 1.0)
  ```
- **Classification:** contract gap (O6.1)
- **Tracking:** owner [#216](https://github.com/mlakolar/ProximalBase.jl/issues/216); related [#239](https://github.com/mlakolar/ProximalBase.jl/issues/239), [#543](https://github.com/mlakolar/ProximalBase.jl/issues/543)
- **Priority:** P3 — caller misuse that the traits already reveal
- **Release impact:** Proposed: candidate for deferral (Q1 settled that this is a gap; the release impact is not decided)
- **Resolution criteria:** the call throws before writing any block

#### LIM-006

- **Summary:** A tuple `Conjugate` writes earlier output blocks before a later block fails to convert to its destination.
- **Affected API:** `prox!`, `value_and_prox!` of `Conjugate(SeparableSum(...))`, including the Fenchel value fallback
- **Conditions:** a later destination cannot hold the result (integer element type, unrecognised array type)
- **Expected:** O6.1 / T1: destination validation before writing; integer destinations rejected
- **Actual:** `InexactError` after block 1 was written (both #371 cases, both oracles)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `InexactError: Int64(0.5)`; block 1 written. At `f9030a64`: 3 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  c = Conjugate(SeparableSum(NormLinf(), NormLinf()))
  prox!(([7.0], Unknown(Int[7, 7])), c, ([2.0], [1.0, 1.0]), 1.0)   # Unknown: #371
  ```
- **Classification:** contract gap (O6.1, T1)
- **Tracking:** owner [#371](https://github.com/mlakolar/ProximalBase.jl/issues/371); related [#543](https://github.com/mlakolar/ProximalBase.jl/issues/543), [#367](https://github.com/mlakolar/ProximalBase.jl/issues/367)
- **Priority:** P2 — partial write for unusual destinations
- **Release impact:** Proposed: candidate for deferral (Q1 settled that this is a gap; the release impact is not decided)
- **Resolution criteria:** both #371 reproducers leave block 1 unchanged

#### LIM-007

- **Summary:** `quadratic_model!` with a nested block mismatch updates earlier sibling models before it raises.
- **Affected API:** `quadratic_model!` of nested `SeparableSum` models
- **Conditions:** nested model with fewer blocks than the function, or a model block of the wrong type
- **Expected:** E1 / O6.1: `DimensionMismatch` (sizes) or `ArgumentError` (types) before any model block is written
- **Actual:** the first sibling's `y` and `weights` are overwritten, then `DimensionMismatch` (or a misleading "no Hessian is known")
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: throws after the first sibling was updated. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  top = SeparableSum(f1, SeparableSum(f2, f3))   # LogisticLoss blocks
  bad = SeparableSum(quadratic_model(f1, β), SeparableSum(quadratic_model(f2, β)))
  quadratic_model!(bad, top, (β, (β, β)))
  ```
- **Classification:** contract gap (E1, O6.1)
- **Tracking:** owner [#495](https://github.com/mlakolar/ProximalBase.jl/issues/495) (group tracker; no specific issue yet); related [#501](https://github.com/mlakolar/ProximalBase.jl/issues/501), [#491](https://github.com/mlakolar/ProximalBase.jl/issues/491)
- **Priority:** P3 — misuse of the in-place model update
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** a nested count or type mismatch leaves every model block unchanged

#### LIM-008

- **Summary:** A dense `Symmetric` destination fails inside Base after part of the output was written.
- **Affected API:** `prox!`, `value_and_prox!` of ordinary functions and `Conjugate` fallbacks
- **Conditions:** `Symmetric(zeros(n, n))` (or `Hermitian`) as the output
- **Expected:** O6.1: a destination type that cannot be written is rejected before writing
- **Actual:** Base's "Cannot set a non-diagonal index in a symmetric matrix" after the diagonal was written (the wrappers `Translate`, `Tilt`, `Regularize` reject first)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `ArgumentError` from Base; `S` changed. At `f9030a64`: 7 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  S = Symmetric(zeros(3, 3)); prox!(S, NormL1(1.0), ones(3, 3), 0.5)
  ```
- **Classification:** contract gap (O6.1, T1)
- **Tracking:** owner [#495](https://github.com/mlakolar/ProximalBase.jl/issues/495) (group tracker; no specific issue yet); related [#288](https://github.com/mlakolar/ProximalBase.jl/issues/288)
- **Priority:** P3 — unsupported destination type
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** a `Symmetric`/`Hermitian` destination raises before writing for every oracle

#### LIM-009

- **Summary:** A fractional result into an integer destination raises `InexactError`, and the sparse route writes earlier entries first.
- **Affected API:** `prox!` into `SparseIterate{Int}` (and dense `Vector{Int}`), `prox_coordinate!` on integer iterates
- **Conditions:** an integer destination and a non-integral result
- **Expected:** T1 / decision 8: an integer destination is rejected for operations with fractional results; O6.1: `ArgumentError` before writing
- **Actual:** `InexactError` (not `ArgumentError`), after the entries before the first inexact one were written, for `SparseIterate{Int}` and dense `Vector{Int}` alike (`prox!(fill(7, 3), NormL1(1.0), [3.0, 2.5, 0.0], 1.0)` leaves `[2, 7, 7]`)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `InexactError: Int64(2.5)`; first entry written. At `f9030a64`: 4 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  si = SparseIterate(Int, 3); prox!(si, NormL1(1.0), [2.0, 3.5, 4.0], 1.0)
  ```
- **Classification:** contract gap (T1, O6.1, decision 8)
- **Tracking:** owner [#492](https://github.com/mlakolar/ProximalBase.jl/issues/492); related [#495](https://github.com/mlakolar/ProximalBase.jl/issues/495)
- **Priority:** P2 — wrong class and partial write for integer destinations
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** fractional operations into integer destinations raise `ArgumentError` before any write

#### LIM-010

- **Summary:** Complex sources written into iterates through `copyto!`, views and reshapes bypass the real-only check and write before failing.
- **Affected API:** `copyto!(x, src)`, `view(x, …) .= src`, `reshape(x, …) .= src` for `SparseIterate` and `AtomIterate`
- **Conditions:** complex source (a nonzero imaginary part fails late; a zero one is accepted)
- **Expected:** S1 and the real-only rule (interface.md): `ArgumentError("… requires real entries")` before writing
- **Actual:** `InexactError` after entry 1 was written; the direct broadcast `x .= src` rejects correctly
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `InexactError`; `s == [1, 0, 0, 0]`. At `f9030a64`: 11 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  s = SparseIterate(4); copyto!(s, ComplexF64[1, 2im, 0, 0])
  ```
- **Classification:** contract gap (S1, O6.1)
- **Tracking:** owner [#364](https://github.com/mlakolar/ProximalBase.jl/issues/364); related [#326](https://github.com/mlakolar/ProximalBase.jl/issues/326), [#365](https://github.com/mlakolar/ProximalBase.jl/issues/365)
- **Priority:** P3 — unsupported input, partial write
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** complex sources through every route raise `ArgumentError` with the iterate unchanged

#### LIM-011

- **Summary:** Complex steps and complex scalars raise `MethodError`, not the documented real-only `ArgumentError`.
- **Affected API:** `prox!`, `value_and_prox!`, `prox_entry`, `prox_coordinate!`, `value_entry` with a complex scalar
- **Conditions:** complex scalar step (a complex array step is rejected correctly)
- **Expected:** interface.md (Precision): a complex step raises `ArgumentError("<Name> requires real entries")` before writing
- **Actual:** `MethodError`; nothing is written
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `MethodError`. At `f9030a64`: 8 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox!(zeros(2), NormL1(1.0), [1.0, 2.0], 1.0 + 0im)
  ```
- **Classification:** contract gap (S1)
- **Tracking:** owner [#365](https://github.com/mlakolar/ProximalBase.jl/issues/365); related [#326](https://github.com/mlakolar/ProximalBase.jl/issues/326), [#364](https://github.com/mlakolar/ProximalBase.jl/issues/364)
- **Priority:** P3 — wrong exception class, no write
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** every scalar entry point raises the real-only `ArgumentError` for a complex step

#### LIM-012

- **Summary:** A complex adjoint product with a `SparseIterate` returns a result, although the docs say a complex operand raises.
- **Affected API:** `A' * x`, `mul!(out, A', x)` with complex `A` and `x::SparseIterate`
- **Conditions:** complex matrix, adjoint form only (`A * x`, `transpose(A) * x` raise correctly)
- **Expected:** iterates.md: a complex operand of an iterate product raises `ArgumentError`
- **Actual:** the generic dense product returns a complex vector; `mul!` into a real output gives `InexactError`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: returns a `ComplexF64` vector. At `f9030a64`: 3 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  xs = SparseIterate(4); xs[2] = 1.0; A = rand(ComplexF64, 4, 4); A' * xs
  ```
- **Classification:** contract gap (S1 (docs))
- **Tracking:** owner [#492](https://github.com/mlakolar/ProximalBase.jl/issues/492) (group tracker; no specific issue yet); related [#364](https://github.com/mlakolar/ProximalBase.jl/issues/364), [#365](https://github.com/mlakolar/ProximalBase.jl/issues/365)
- **Priority:** P3 — the result is correct; only the documented rejection is missing
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** `A' * xs` and `mul!(out, A', xs)` raise `ArgumentError`, or the docs state the exception

### Steps (P1–P3)

#### LIM-013

- **Summary:** Most functions accept negative and NaN steps: a negative step silently expands; a NaN step gives NaN or an arbitrary result.
- **Affected API:** `prox!`, `value_and_prox!`, `prox_entry`, `prox_coordinate!` of most leaves (`NormL1`, `SqrNormL2`, `IndBox`, `IndHyperplane`, `IndHalfspace`, `IndHyperslab`, `IndIsotonic`, `IndSOC`, `IndBallGroupL0`, `NuclearNorm`, `SLOPE`, `TotalVariation1D`, …) and therefore of block sums
- **Conditions:** scalar step `γ < 0` or `γ = NaN` (the balls, sphere and simplex reject since #522)
- **Expected:** P1 and decision 25: `ArgumentError` before anything is written, in every route
- **Actual:** `prox(NormL1(1.0), [0.5, 0.5], -1.0) == [1.0, 1.0]`; `SqrNormL2` gives `[-Inf, -Inf]`; at a NaN step `NormL1`, `SqrNormL2` and `SLOPE` give NaN, the indicators (`IndBox`, `IndHyperplane`, `IndHalfspace`, `IndHyperslab`, `IndIsotonic`, `IndSOC`, `IndBallGroupL0`) return their projection, `TotalVariation1D` returns its input and `NuclearNorm` returns zeros; every named leaf accepts both steps and writes, through `prox!`, `value_and_prox!`, `prox_entry`, `prox_coordinate!` and block sums
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[1.0, 1.0]`; `[1.0, 0.0]`; `1.0`. At `f9030a64`: 44 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(NormL1(1.0), [0.5, 0.5], -1.0)
  prox(IndBox(-1.0, 1.0), [3.0, 0.0], NaN)
  prox_entry(NormL1(1.0), 0.5, 1, -1.0)
  ```
- **Classification:** contract gap (P1, P2, O6.1, decision 25)
- **Tracking:** owner [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311); related [#486](https://github.com/mlakolar/ProximalBase.jl/issues/486), [#290](https://github.com/mlakolar/ProximalBase.jl/issues/290), [#522](https://github.com/mlakolar/ProximalBase.jl/issues/522), [#543](https://github.com/mlakolar/ProximalBase.jl/issues/543)
- **Priority:** P1 — a sign error in a solver's step gives silently wrong iterates
- **Release impact:** Proposed: blocks 0.6
- **Resolution criteria:** for every exported proximable leaf, `prox!(y, f, x, -1.0)` and `NaN` raise `ArgumentError` with `y` unchanged, through every route and inside a block sum

#### LIM-014

- **Summary:** Some functions reject a negative step only for some argument values, or with `DomainError`, and write earlier blocks first.
- **Affected API:** `BernoulliLoss`, `CubeNormL2` (value-dependent); `NegLog`, `NegEntropy`, `KLDivergence`, `LogDetLoss`, `PointwiseMinimum` (wrong class)
- **Conditions:** `γ < 0`; `BernoulliLoss` only for entries with a nonzero weight (a zero-weight entry is accepted and written first), `CubeNormL2` only for a nonzero argument
- **Expected:** P1: `ArgumentError` before writing, independent of the argument values
- **Actual:** `DomainError` (from `sqrt` for `BernoulliLoss` and `CubeNormL2`); `LogDetLoss` raises `DomainError` at a NaN step too, while `PointwiseMinimum` rejects NaN with a correct `ArgumentError` and only its negative step is a `DomainError`; `BernoulliLoss([0, 1], [0, 1])` at `[0.2, 0.7]` writes `y[1] = 0.2` before raising; in a block sum, `BernoulliLoss` and `CubeNormL2` raise after block 1 was written (block 1 is written because `SqrNormL2` accepts `γ = -1`, LIM-013), whereas `NegLog` and the other wrong-class functions fail in the preflight with nothing written
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `DomainError`; block 1 written. At `f9030a64`: 19 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  s = SeparableSum(SqrNormL2(2.0), BernoulliLoss([0.0, 1.0]))
  prox!(([7.0], [7.0, 7.0]), s, ([3.0], [0.2, 0.7]), -1.0)
  ```
- **Classification:** contract gap (P1)
- **Tracking:** owner [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311); related [#486](https://github.com/mlakolar/ProximalBase.jl/issues/486), [#543](https://github.com/mlakolar/ProximalBase.jl/issues/543)
- **Priority:** P2 — value-dependent validation and wrong exception class
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** these functions raise `ArgumentError` for every negative or NaN step before writing, standalone and in block sums

#### LIM-015

- **Summary:** Some functions return NaN at an infinite step that they do not name as an endpoint, instead of a limit or a rejection.
- **Affected API:** `SqrHingeLoss`, `SparseGroupNormL2(0, 0, groups)` (prox at `γ = Inf`); `SqrNormL2(0)` at `γ = Inf` is a zero-level case, LIM-053
- **Conditions:** `γ = Inf`
- **Expected:** P1 and decision 26: the named limit, or `ArgumentError` before writing; never NaN
- **Actual:** `prox(SqrHingeLoss(), [0, 2], Inf) == [NaN, 2]` (`HingeLoss` gives `[1, 2]`); the zero-level `SparseGroupNormL2` gives `[NaN, NaN]` (`GroupNormL2(0)` gives the identity)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[NaN, 2.0]`; `[NaN, NaN]`. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(SqrHingeLoss(), [0.0, 2.0], Inf)
  prox(SparseGroupNormL2(0.0, 0.0, [[1], [2]]), [1.0, 2.0], Inf)
  ```
- **Classification:** contract gap (P1, E4, decision 26)
- **Tracking:** owner [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311); related [#486](https://github.com/mlakolar/ProximalBase.jl/issues/486)
- **Priority:** P2 — literal infinite steps on a few functions
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** each call returns its limit (`[1, 2]`; the identity) or raises before writing

#### LIM-016

- **Summary:** Subnormal and huge scalar steps on `LinearConstrained`, and diagonal steps that `W` cannot represent on linear sets, give silently infeasible or `-Inf` results.
- **Affected API:** `Constrained(f, IndHyperplane/IndHalfspace/IndHyperslab)` with scalar steps; `IndHyperplane`, `IndHalfspace`, `IndHyperslab` with array steps
- **Conditions:** scalar `γ = 1e-310` or `5e-324`, or `γ = 1e300` and `1e308`; equal array steps `[1e-310, 1e-310]` or `[1e308, 1e308]`; `Float64` steps `[1e-50, 2e-50]` in a `Float32` call (the P3 example)
- **Expected:** P3 / decision 27: a retained scale times ratios, so the projection does not depend on the scale; positive finite steps are valid (P1)
- **Actual:** `Constrained(IndBox(-1,1), IndHyperplane(1,1))` at `γ = 1e-310` returns a point off the hyperplane; `Constrained(NormL1(1), IndHyperplane(ones(3),1))` returns `[-Inf, -Inf, -Inf]`; equal steps `1e308` return `x` unchanged; the P3 example gives `Float32[-Inf, -Inf]` for `IndHyperplane` and `IndHalfspace` (`IndHyperslab` returns the projection there, but fails at equal steps `1e-310` and `1e308`); no error. Found while rechecking: at scalar `γ = 1e300`, after a call at `γ = 1` on the same object (the search starts from the multiplier of the last call), `Constrained(IndBox(-1,1), IndHyperplane(1,1))` returns `[1, 1, 1]`, off the hyperplane (a fresh object returns the projection); at `γ = 1e308` it raises "the IndHyperplane misses the domain of IndBox" for the feasible set
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[0.3, -1.0, 1.0]` (sum 0.3); `[3.0, 3.0]`; `Float32[-Inf, -Inf]`. At `f9030a64`: 15 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(Constrained(IndBox(-1.0, 1.0), IndHyperplane(1.0, 1.0)), [0.3, -1.2, 2.0], 1e-310)
  prox(IndHyperplane([1.0, 1.0], 1.0), [3.0, 3.0], [1e308, 1e308])
  prox(IndHyperplane(Float32[1, 1], 1f0), Float32[3, 3], [1e-50, 2e-50])
  ```
- **Classification:** contract gap (P3, P1, R1, decision 27)
- **Tracking:** owner [#257](https://github.com/mlakolar/ProximalBase.jl/issues/257); related [#299](https://github.com/mlakolar/ProximalBase.jl/issues/299), [#475](https://github.com/mlakolar/ProximalBase.jl/issues/475), [#488](https://github.com/mlakolar/ProximalBase.jl/issues/488)
- **Priority:** P1 — silently infeasible or infinite results for valid steps, including the contract's own P3 example
- **Release impact:** Proposed: blocks 0.6
- **Resolution criteria:** equal array steps `s·[1, …, 1]` agree with the scalar step for `s ∈ {1e-320, 1e-310, 1e300, 1e308}`; the P3 example and the subnormal `LinearConstrained` case return the projection

### Searches (A3)

#### LIM-017

- **Summary:** Multiplier searches accept on a rounding-scale test that does not certify the stored output, and never raise `PrecisionError`.
- **Affected API:** `Constrained(f, IndHyperplane/IndHalfspace/IndHyperslab/IndSimplex)` searches
- **Conditions:** `Float32` (and `Float64`) with many entries and signed normals
- **Expected:** A3 and decision 18: certify the stored output outside a documented band; inside it refine or raise `PrecisionError` before writing
- **Actual:** the search stops at `|φ| ≤ 2eps(T)·m` or after 100 secants with no certificate; on #151's case (`n = 20000`, `Float32`, generated deterministically) the result differs from the `Float64` solve of the same data by 782 `eps(Float32)` on Julia 1.13 and 1,330 on 1.10 (their `libm` differs), with no exception
- **Evidence:** at `920ebece`, on the audit's random draw of #151's case: relative difference 2.1e-4 (Julia 1.13.3) and 8.9e-5 (1.10.12), no exception. At `f9030a64`, on deterministic data: 782 `eps(Float32)` (1.13.3) and 1,330 (1.10.12), no exception; the case prints `PRESENT` on both.
  ```julia
  # #151's table case with deterministic data (_A_data151 in reproduce-active.jl):
  # Constrained(IndBox(0f0, 1f0), IndHyperplane(a, b)) at n = 20000, step 0.7f0, compared with Float64
  ```
- **Classification:** contract gap (A3, A1, O6.3)
- **Tracking:** owner [#151](https://github.com/mlakolar/ProximalBase.jl/issues/151); related [#348](https://github.com/mlakolar/ProximalBase.jl/issues/348), [#488](https://github.com/mlakolar/ProximalBase.jl/issues/488), [#510](https://github.com/mlakolar/ProximalBase.jl/issues/510)
- **Priority:** P2 — uncertified decisions on ordinary `Float32` inputs; residuals stay small
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** a `Float32` search that cannot be certified raises `PrecisionError`; certified outputs meet a stated residual bound on #151's table (A3 states no number yet; the reproducer's 16 `eps(Float32)` stands in for it)

#### LIM-018

- **Summary:** A linear-set search reports "the set misses the domain" for a feasible problem at huge inputs or one dominant per-entry step.
- **Affected API:** `Constrained(f, IndHyperplane/IndHyperslab)` multiplier search
- **Conditions:** `x = [1e300, 1e300]` or `fill(1e20, 4)` with `IndBox(0, 1)`; per-entry steps with one `γᵢaᵢ²` larger than the rest by about `2^120`
- **Expected:** A3 and O6.2: no infeasibility message for a feasible set; certify or raise `PrecisionError`
- **Actual:** `ArgumentError: the IndHyperplane misses the domain of IndBox` (and of `SqrNormL2` in #475's case, and of `NormL1` for per-entry steps `[1e300, 1]`)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `ArgumentError` (false infeasibility). At `f9030a64`: 10 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(Constrained(IndBox(0.0, 1.0), IndHyperplane([1.0, 1.0], 1.0)), [1e300, 1e300], 1.0)
  ```
- **Classification:** contract gap (A3, O6.2)
- **Tracking:** owner [#475](https://github.com/mlakolar/ProximalBase.jl/issues/475); related [#323](https://github.com/mlakolar/ProximalBase.jl/issues/323), [#257](https://github.com/mlakolar/ProximalBase.jl/issues/257), [#488](https://github.com/mlakolar/ProximalBase.jl/issues/488)
- **Priority:** P2 — wrong exception on a feasible problem
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer and #475's table return a point with `⟨a, z⟩ = b`, or raise `PrecisionError`

#### LIM-019

- **Summary:** The ridge-plus-hyperplane search reports a spurious "misses the domain" once `γλ` is large.
- **Affected API:** `Constrained(SqrNormL2/Regularize, IndHyperplane)`
- **Conditions:** `γλ` from about `3e32` (`Float64`; `1e32` passes) or `3e25` (`Float32`; `1e25` passes), well before `1/(1 + γλ)` underflows; #510's own example (`λ = 2^30`, `γ = 2^100`, `Float32`)
- **Expected:** A3 and O6.2: a feasible constraint is not reported as missing the domain
- **Actual:** `ArgumentError: the IndHyperplane misses the domain of SqrNormL2`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `ArgumentError` (exact answer `[2^-30, 2^-30]`). At `f9030a64`: 4 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(Constrained(SqrNormL2(2f0^30), IndHyperplane(1f0, 2f0^-29)), zeros(Float32, 2), 2f0^100)
  ```
- **Classification:** contract gap (A3, O6.2)
- **Tracking:** owner [#510](https://github.com/mlakolar/ProximalBase.jl/issues/510); related [#475](https://github.com/mlakolar/ProximalBase.jl/issues/475), [#476](https://github.com/mlakolar/ProximalBase.jl/issues/476), [#507](https://github.com/mlakolar/ProximalBase.jl/issues/507)
- **Priority:** P2 — spurious infeasibility at extreme steps (loud)
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the #510 example returns `2^-30` per entry, or raises `PrecisionError` inside A3's band

#### LIM-020

- **Summary:** `IndHyperplane` and `IndHalfspace` round the projection coefficient to a narrow output before multiplying the normal.
- **Affected API:** `prox!` of `IndHyperplane`, `IndHalfspace` (scalar and array steps); `IndHyperslab` is fixed
- **Conditions:** `Float32` output, normal entry `2^100`, argument `2^-100`
- **Expected:** T3/T4 and R2: the coefficient is formed in the model precision; the exact result is `Float32[0]`
- **Actual:** returns the argument unchanged
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `Float32[7.88861f-31]`. At `f9030a64`: 4 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox!(zeros(Float32, 1), IndHyperplane([2.0^100], 0.0), Float32[2.0^-100], 1.0)
  ```
- **Classification:** contract gap (T3, T4, R2)
- **Tracking:** owner [#218](https://github.com/mlakolar/ProximalBase.jl/issues/218); related [#353](https://github.com/mlakolar/ProximalBase.jl/issues/353), [#362](https://github.com/mlakolar/ProximalBase.jl/issues/362)
- **Priority:** P2 — wrong value at extreme but valid scales
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer returns `Float32[0]` for hyperplane and halfspace, scalar and array steps

#### LIM-021

- **Summary:** `IndHyperplane` and `IndHalfspace` projections return `-Inf` when `⟨a, x⟩` overflows although the projection is representable.
- **Affected API:** `prox!` of `IndHyperplane`, `IndHalfspace`
- **Conditions:** finite entries near `floatmax` so that `⟨a, x⟩` (or `⟨a, x⟩ − b`) overflows
- **Expected:** R2: closed-form projection entries are returned whenever representable
- **Actual:** `IndHalfspace([1, 1], 0)` at `[M, M]` gives `[-Inf, -Inf]` (exact `[0, 0]`); no error
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[-Inf, -Inf]`. At `f9030a64`: 4 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  M = floatmax(Float64); prox(IndHalfspace([1.0, 1.0], 0.0), [M, M], 1.0)
  ```
- **Classification:** contract gap (R2, R4)
- **Tracking:** owner [#563](https://github.com/mlakolar/ProximalBase.jl/issues/563) (opened from this audit); related [#488](https://github.com/mlakolar/ProximalBase.jl/issues/488) (G5), [#361](https://github.com/mlakolar/ProximalBase.jl/issues/361), [#218](https://github.com/mlakolar/ProximalBase.jl/issues/218), [#257](https://github.com/mlakolar/ProximalBase.jl/issues/257)
- **Priority:** P2 — silently wrong output at extreme finite inputs
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer returns `[0, 0]`; `IndHyperplane([1, 1], 1e308)` at `[M, M]` returns `[5e307, 5e307]`

### Range (R1–R4)

#### LIM-022

- **Summary:** Scalar kernels and losses form an intermediate that overflows or underflows before the weight, losing a representable value.
- **Affected API:** values of `Berhu`, `CappedL1`, `HingeLoss`, `CheckLoss`, `LogPenalty`, `MCP`, `SCAD`, `NegLog`, the GLM link losses, and `SingularValue`/`Spectral` of `MCP`/`SCAD`
- **Conditions:** finite data and weights whose weighted value is normal
- **Expected:** R1: a normal value is returned despite an intermediate leaving the range
- **Actual:** `MCP(1e154, 3)([2e154])` is NaN (exact ≈ 1.5e308); `Berhu(1e-300, 1)([1e200])` is `Inf` (5e99); `CheckLoss(1e-200, [1e300])([1e-200])` is `0`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `NaN`. At `f9030a64`: 13 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  MCP(1e154, 3.0)([2e154])
  ```
- **Classification:** contract gap (R1)
- **Tracking:** owner [#356](https://github.com/mlakolar/ProximalBase.jl/issues/356); related [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283)
- **Priority:** P2 — Inf, NaN or 0 instead of a representable value
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** every row of #356's table returns its stated value

#### LIM-023

- **Summary:** `Translate` and `Precompose` of a norm do not recover an overflowing transformed argument through homogeneity; the contract's R1 example fails.
- **Affected API:** values of `Translate(NormL1/NormL2/NormLinf, b)`, `Precompose(norm, a)` (scalar `a`, array `a`, and a linear map `A` with `μ`)
- **Conditions:** `x − b` or `a·x` overflows while the value is representable
- **Expected:** R1 (wrappers) and decision 45: norms recover through homogeneity; `Translate(NormL1(0.25), [-M])([M]) == M/2` (W1)
- **Actual:** `Inf`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `Inf` (contract: `M/2`). At `f9030a64`: 7 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  M = floatmax(Float64); Translate(NormL1(0.25), [-M])([M])
  ```
- **Classification:** contract gap (R1, decision 45)
- **Tracking:** owner [#357](https://github.com/mlakolar/ProximalBase.jl/issues/357); related [#490](https://github.com/mlakolar/ProximalBase.jl/issues/490), [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283)
- **Priority:** P2 — a guarantee stated by example on the contract page fails
- **Release impact:** Not yet decided (Q2)
- **Resolution criteria:** W1 returns `M/2`; `Precompose(NormL1(0.25), 4.0)([M]) == M`

#### LIM-024

- **Summary:** Values of sums and calculus wrappers lose range: partial sums, dots and inner values overflow before the result is formed.
- **Affected API:** values of `Sum`, `SeparableSum`, `SliceSum`, `Tilt`, `MoreauEnvelope`, `Conjugate`, `Scaled` (inner function without a foldable level), `LinearImage`, `GroupMCP`
- **Conditions:** representable result with an out-of-range intermediate
- **Expected:** R1 and decision 45: the private scaled-value mechanism recovers inner values
- **Actual:** `Scaled(SqrNormL2(), 1e-200)([1e200])` is `Inf` (5e199); `Sum` of three `LinearFunction`s with constants `M, M, −M` is `Inf` (M); `Conjugate(IndBox(1, 2))([0.75M, −M])` is `Inf` (0.5M); the same holds for `SeparableSum`, `SliceSum`, `Tilt`, `MoreauEnvelope`, `Conjugate(Scaled(IndBallL2(), s))`, `GroupMCP` and `LinearImage` (16 cases in `reproduce-active.jl`). Since #549, `numerics.md` ("Nested scales", under R1) states that a function that folds no level (`SqrNormL2`, a loss, a `Tilt`) loses an out-of-range intermediate inside `Scaled`, a single `Scaled` included. That documents the `Scaled` rows of this gap; it does not approve them (no **Decided** line; S4). One `LinearImage` row (`LinearImage(NormL1(1e-300), [0.5;;], 0.25)` at `[M]`) has an overflowing transformed argument, which R1's wrapper rule lists as not covered (decision 45, LIM-083); it stays here until the maintainer reads it one way or the other
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `Inf`. At `f9030a64`: 16 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  Scaled(SqrNormL2(), 1e-200)([1e200])
  ```
- **Classification:** contract gap (R1, decision 45)
- **Tracking:** owner [#359](https://github.com/mlakolar/ProximalBase.jl/issues/359); related [#567](https://github.com/mlakolar/ProximalBase.jl/issues/567) (the `Scaled` rows), [#568](https://github.com/mlakolar/ProximalBase.jl/issues/568), [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283), [#490](https://github.com/mlakolar/ProximalBase.jl/issues/490)
- **Priority:** P2 — Inf or NaN instead of a representable value
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** every row of #359's table returns its stated value

#### LIM-025

- **Summary:** `value_and_prox!` of `Scaled` returns `0` when the inner value underflows, although the scaled value of the stored output is normal.
- **Affected API:** `value_and_prox!` of `Scaled(f, α)`
- **Conditions:** `Scaled(NormL1(2^-1000), 2^1000)` at `[2^-100]`, step `2^-1000`
- **Expected:** O2: the value at the stored output, with no false zero
- **Actual:** returns `0.0` while `f(y) == 2^-100`; unchanged by #549, and not mentioned in its "Nested scales" paragraph (O2 forbids the false zero)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `(0.0, 7.888609052210118e-31)`. At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  f = Scaled(NormL1(2.0^-1000), 2.0^1000); y = zeros(1)
  value_and_prox!(y, f, [2.0^-100], 2.0^-1000), f(y)
  ```
- **Classification:** contract gap (O2, R1)
- **Tracking:** owner [#359](https://github.com/mlakolar/ProximalBase.jl/issues/359); related [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283)
- **Priority:** P2 — false zero value
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** `value_and_prox!` returns `f(y)` for the reproducer

#### LIM-026

- **Summary:** `NormL2`, `GroupNormL2` and `DistL2` round a subnormal norm before a large weight; `NegLog` and `Conjugate(SqrHingeLoss)` overflow in signed reductions.
- **Affected API:** values of `NormL2`, `GroupNormL2`, `DistL2`, `NegLog`, `Conjugate(SqrHingeLoss)`
- **Conditions:** subnormal entries with a weight near `1/floatmin`; signed sums near `floatmax`
- **Expected:** R1
- **Actual:** `NormL2(1e300)([s, s])` with `s = nextfloat(0.0)` is `4.94e-24` (exact `6.99e-24`, 29% low)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `4.940656458412466e-24`. At `f9030a64`: 5 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  NormL2(1e300)([nextfloat(0.0), nextfloat(0.0)])
  ```
- **Classification:** contract gap (R1)
- **Tracking:** owner [#360](https://github.com/mlakolar/ProximalBase.jl/issues/360); related [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283)
- **Priority:** P2 — 29% error or `Inf` for representable values
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the B17 and B19 rows of #360 return their values

#### LIM-027

- **Summary:** Proximal outputs of `Berhu`, `HingeLoss` and `NormL1Fused` overflow at finite inputs with representable results.
- **Affected API:** `prox!` of `Berhu`, `HingeLoss`, `NormL1Fused` (including its pooling preprocessing)
- **Conditions:** entries near `floatmax` or large weights
- **Expected:** R2
- **Actual:** `prox(HingeLoss([2.0]), [M], 1)` is `[Inf]` (exact `[M]`); `prox(NormL1Fused(0, 1), [M, M, M], 1e308)` is all NaN
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[Inf]`. At `f9030a64`: 4 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(HingeLoss([2.0]), [floatmax(Float64)], 1.0)
  ```
- **Classification:** contract gap (R2)
- **Tracking:** owner [#361](https://github.com/mlakolar/ProximalBase.jl/issues/361); related [#554](https://github.com/mlakolar/ProximalBase.jl/issues/554)
- **Priority:** P2 — wrong output entries at extreme finite inputs
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** each row of #361 returns its output

#### LIM-028

- **Summary:** Exponential-cone projections return zeros and a saturated entry when one exact output component overflows.
- **Affected API:** `prox!` of `IndExpPrimal`, `IndExpDual`
- **Conditions:** finite argument near `floatmax` whose exact projection has one component above `floatmax`
- **Expected:** R4 (overflow only in the entries whose exact value overflows) or `PrecisionError` before writing
- **Actual:** `prox(IndExpDual(), [M, −M, M])` is `[-0, -0, M]` (scale-consistent reference `[-2.6e307, -7.8e307, Inf]`); representable cases are correct
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[-0.0, -0.0, 1.7976931348623157e308]`. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(IndExpDual(), [M, -M, M], 1.0)
  ```
- **Classification:** contract gap (R4, R2)
- **Tracking:** owner [#361](https://github.com/mlakolar/ProximalBase.jl/issues/361); related [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485)
- **Priority:** P3 — only inputs near `floatmax`
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer returns the reference entries or raises `PrecisionError` before writing (add the row to #361)

#### LIM-029

- **Summary:** The weighted `NormL1` value loses products `wᵢ|xᵢ|` that underflow although their total is normal.
- **Affected API:** value of `NormL1(λ, w)`
- **Conditions:** `Float32`, `w = 2^-126`, `x ≈ 2^-9`, `n = 512`
- **Expected:** R1
- **Actual:** `1.1754944f-38` (exact `1.1755302f-38`, 256 ulps low)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `1.1754944f-38`. At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  NormL1(1f0, fill(2f0^-126, 512))(fill(Float32(2.0^-9 + 2.0^-24), 512))
  ```
- **Classification:** contract gap (R1)
- **Tracking:** owner [#373](https://github.com/mlakolar/ProximalBase.jl/issues/373); related [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283)
- **Priority:** P2 — 2e-5 relative error at a normal value
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer is within 4 ulps of the exact value

#### LIM-030

- **Summary:** Values of `SingularValue(f)`, `Spectral(f)` and `Conjugate(SingularValue(f))` overflow at a finite matrix whose spectrum overflows.
- **Affected API:** standalone values of the spectral wrappers (e.g. `SingularValue(NormL1(λ))`, which should equal `NuclearNorm(λ)`)
- **Conditions:** `X = fill(M, 2, 2)` (`σ₁ = 2M`)
- **Expected:** R1 and decision 45: a representable value is returned for built-in inner functions
- **Actual:** `SingularValue(NormL1(0.25))(X)` is `Inf` while `NuclearNorm(0.25)(X) == M/2`; `value_and_prox!` returns the right value
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `(Inf, 8.98847e307)`. At `f9030a64`: 8 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  X = fill(floatmax(Float64), 2, 2); SingularValue(NormL1(0.25))(X), NuclearNorm(0.25)(X)
  ```
- **Classification:** contract gap (R1, decision 45)
- **Tracking:** owner [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283); related [#384](https://github.com/mlakolar/ProximalBase.jl/issues/384), [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489), [#359](https://github.com/mlakolar/ProximalBase.jl/issues/359)
- **Priority:** P2 — documented equivalence fails at extreme scale
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** `SingularValue(NormL1(0.25))(X) == NuclearNorm(0.25)(X)`

#### LIM-031

- **Summary:** `QuadraticFunction` and `KronQuadratic` values lose accuracy when an intermediate `(Ax)ᵢ` underflows while a large `xᵢ` multiplies it.
- **Affected API:** values of `QuadraticFunction`, `KronQuadratic`
- **Conditions:** `x = [1.23·2^1000, 1.11·2^-60]`, `A = [0 a; a 0]`, `a = 1.37·2^-970`
- **Expected:** R1
- **Actual:** relative error `7.8e-15` (about 35 eps), the same for `KronQuadratic([0 a; a 0], [1;;])` at `reshape(x, 2, 1)`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: relative error `7.79e-15` against BigFloat. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  a = 1.37 * 2.0^-970; QuadraticFunction([0 a; a 0])([1.23 * 2.0^1000, 1.11 * 2.0^-60])
  ```
- **Classification:** contract gap (R1)
- **Tracking:** owner [#520](https://github.com/mlakolar/ProximalBase.jl/issues/520); related [#536](https://github.com/mlakolar/ProximalBase.jl/issues/536), [#358](https://github.com/mlakolar/ProximalBase.jl/issues/358)
- **Priority:** P3 — contrived scale mix
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** relative error at most 4 eps

#### LIM-032

- **Summary:** `GLMLoss` values do not recover observation weights and linear predictors that leave the range.
- **Affected API:** values of `PoissonLoss`, `LogisticLoss`, squared-hinge GLMs
- **Conditions:** observation weights near `1e±300`; predictors that cancel near `floatmax`
- **Expected:** R1
- **Actual:** `PoissonLoss([0], ones(1, 1); weights = [1e-300])([1000])` is `Inf` (exact `1.97e134`); a cancelling predictor gives NaN (exact 1)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `Inf`. At `f9030a64`: 4 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  PoissonLoss([0.0], ones(1, 1); weights = [1e-300])([1000.0])
  ```
- **Classification:** contract gap (R1)
- **Tracking:** owner [#358](https://github.com/mlakolar/ProximalBase.jl/issues/358); related [#539](https://github.com/mlakolar/ProximalBase.jl/issues/539), [#541](https://github.com/mlakolar/ProximalBase.jl/issues/541)
- **Priority:** P2 — Inf, NaN or 0 for representable values
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the B11 rows of #358 return their values

#### LIM-033

- **Summary:** `GroupMCP` and `GroupSCAD` proximal outputs are NaN when the group norm overflows the argument type.
- **Affected API:** `prox!` of `GroupMCP`, `GroupSCAD`
- **Conditions:** `Float32[3e38, 3e38, 3e38]` (or `[M, M]` in `Float64`)
- **Expected:** R2 (`GroupNormL2` recovers)
- **Actual:** `[NaN, NaN, 3e38]`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[NaN, NaN, 3.0e38]`. At `f9030a64`: 5 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox!(zeros(3), GroupMCP(1.0, [1:2, 3:3], [1.0, 1.0]; a = 3), Float32[3e38, 3e38, 3e38], 1.0)
  ```
- **Classification:** contract gap (R2)
- **Tracking:** owner [#534](https://github.com/mlakolar/ProximalBase.jl/issues/534); related [#511](https://github.com/mlakolar/ProximalBase.jl/issues/511), [#359](https://github.com/mlakolar/ProximalBase.jl/issues/359)
- **Priority:** P2 — NaN from finite input
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer matches its BigFloat reference

#### LIM-035

- **Summary:** The Poisson quadratic model holds an infinite weight and a NaN working response at `η ≥ 710`.
- **Affected API:** `quadratic_model`, `quadratic_model!` of `PoissonLoss`
- **Conditions:** `η = 710` (effective coefficient `exp(710)/8 ≈ 2.79e307` with weight `1/8`)
- **Expected:** S2 and decision 43: a finite effective coefficient, or `PrecisionError` before `q` is written
- **Actual:** `weights == [Inf]`, `y == [NaN]`, no exception, from `quadratic_model` and `quadratic_model!` (which overwrites `q`); with the default weight the coefficient `exp(710)` itself exceeds `floatmax`, so there only `PrecisionError` is correct
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `([Inf], [NaN])`. At `f9030a64`: 3 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  q = quadratic_model(PoissonLoss([1.0], ones(1, 1); weights = [1/8]), [710.0]); q.weights, q.y
  ```
- **Classification:** contract gap (S2, O6.3, decision 43)
- **Tracking:** owner [#317](https://github.com/mlakolar/ProximalBase.jl/issues/317); related [#544](https://github.com/mlakolar/ProximalBase.jl/issues/544), [#541](https://github.com/mlakolar/ProximalBase.jl/issues/541), [#548](https://github.com/mlakolar/ProximalBase.jl/issues/548)
- **Priority:** P2 — a non-finite model returned silently
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer's weight equals `exp(big(710))/8` within 4 eps, or `PrecisionError`

#### LIM-036

- **Summary:** `Constrained(NormL1, IndSphereL2)` raises when `γλw` overflows although the projection is representable.
- **Affected API:** `prox!` of `Constrained(NormL1, IndSphereL2)`
- **Conditions:** finite `γ`, `λ`, `w` whose product exceeds `floatmax`
- **Expected:** R3: a combined parameter that overflows is computed as if unrounded
- **Actual:** `ArgumentError: sphere thresholds must be finite` (the ball version returns `[0, 0]`; a test pins the rejection, `test/indicator_steps.jl:207`)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `ArgumentError`. At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(Constrained(NormL1(1e300), IndSphereL2(1.0)), [1.0, 2.0], 1e300)
  ```
- **Classification:** contract gap (R3)
- **Tracking:** owner [#484](https://github.com/mlakolar/ProximalBase.jl/issues/484) (group tracker; no specific issue yet); related [#469](https://github.com/mlakolar/ProximalBase.jl/issues/469), [#545](https://github.com/mlakolar/ProximalBase.jl/issues/545), [#557](https://github.com/mlakolar/ProximalBase.jl/issues/557)
- **Priority:** P2 — a representable case is rejected
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer returns the projection, or a decision records the exclusion

#### LIM-037

- **Summary:** `deriv2` of `Smoothed(φ, γ)` is a central finite difference with about `√eps` accuracy, which its docstring does not state.
- **Affected API:** `ProximalBase.deriv2(::Smoothed, …)` and the second-order oracles built on it
- **Conditions:** `φ` whose scalar proximal map is not piecewise linear (e.g. `NegEntropy`)
- **Expected:** A1: a closed form with R2 accuracy, or a stated error scale
- **Actual:** relative error `6.3e-9` (2.8e7 eps) in `Float64`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: relative error ≈ 6e-9. At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  ProximalBase.deriv2(ProximalBase.Smoothed(NegEntropy(1.0), 0.5), 0.7, 0.0)   # vs BigFloat
  ```
- **Classification:** contract gap (A1)
- **Tracking:** owner [#491](https://github.com/mlakolar/ProximalBase.jl/issues/491) (group tracker; no specific issue yet); related [#366](https://github.com/mlakolar/ProximalBase.jl/issues/366)
- **Priority:** P3 — accuracy of a curvature used for Newton directions
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** a closed form, or a documented error scale with a test at that scale

### Precision (T3–T6, O2)

#### LIM-038

- **Summary:** A function that holds narrow data, evaluated at a wider argument, computes in the data's precision; the contract's T3 example fails.
- **Affected API:** `QuadraticFunction`, `LeastSquaresLoss`, `KronQuadratic`, `IndAffine`, `LogDetLoss` (values, gradients, proximal outputs)
- **Conditions:** `Float32` data, `Float64` argument or destination
- **Expected:** T3, T4 and decision 6: the model precision includes the argument's; `QuadraticFunction(Float32[1;;], Float32[-1])` at `1 + 2^-28` into a `Float32` gradient gives `2^-28`
- **Actual:** `Float32`-level accuracy: `QuadraticFunction(Float32[2 0.3; 0.3 2], 0)` at `[1 + 1e-9, 2 + 1e-9]` differs from the `Float64`-data function by `2.35e-7` (relative `4.2e-8`), `LogDetLoss(Float32[2 0.1; 0.1 3], 1f0)` `value_and_prox!` at `[2 0.5; 0.5 3]`, step `0.5`, by `1.62e-7` in the output and `3.6e-7` in the value, and the `IndAffine` prox residual is `2.4e-8` (`5.6e-17` with `Float64` data); the T3 example gives `0`; for `QuadraticFunction` the standalone value and `value_and_gradient!` disagree (the earlier figures 2e-8 and 1.6e-7 came from random data)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `Float32[0.0]` (contract: `2^-28`). At `f9030a64`: 12 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  gradient!(zeros(Float32, 1), QuadraticFunction(Float32[1;;], Float32[-1]), [1 + 2.0^-28])
  ```
- **Classification:** contract gap (T3, T4, O1, decision 6)
- **Tracking:** owner [#487](https://github.com/mlakolar/ProximalBase.jl/issues/487); related [#358](https://github.com/mlakolar/ProximalBase.jl/issues/358), [#513](https://github.com/mlakolar/ProximalBase.jl/issues/513)
- **Priority:** P2 — `Float64` callers get `Float32`-level values
- **Release impact:** Not yet decided (Q2)
- **Resolution criteria:** with `Float32` data at a `Float64` argument these functions agree with the `Float64`-data function to a few `eps(Float64)`; the T3 example returns `2^-28`

#### LIM-039

- **Summary:** Separable kernels with narrow parameters and a wider destination combine the parameters in the narrow type; the contract's V10 row fails.
- **Affected API:** `prox!` of `NormL1` and other separable kernels
- **Conditions:** `Float32` argument and parameters, `Float64` destination
- **Expected:** T4: computed in `D` from exactly widened inputs; V10 gives `2^-46`
- **Actual:** `[0.0]`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[0.0]` (contract: `2^-46`). At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox!(zeros(1), NormL1(1 + 2f0^-23), Float32[1], 1 - 2f0^-23)
  ```
- **Classification:** contract gap (T4, T3)
- **Tracking:** owner [#554](https://github.com/mlakolar/ProximalBase.jl/issues/554); related [#519](https://github.com/mlakolar/ProximalBase.jl/issues/519), [#411](https://github.com/mlakolar/ProximalBase.jl/issues/411)
- **Priority:** P2 — a contract table row is violated
- **Release impact:** Not yet decided (Q2)
- **Resolution criteria:** the V10 call returns `[2^-46]`

#### LIM-040

- **Summary:** Threshold and coefficient products are formed after rounding the factors to the argument type in several families.
- **Affected API:** `CheckLoss`, `TotalVariation1D`, `FusedLasso` (and the other sites listed in #519)
- **Conditions:** `Float32` factors whose product overflows `Float32`, `Float64` argument
- **Expected:** T3: a combined parameter is formed in `promote_type(W, factor types)` and rounded once
- **Actual:** `prox(TotalVariation1D(2f0^80), [2.0^161, 1.0], 2f0^80)` is `[NaN, NaN]`; `CheckLoss` gives `[0, 0]`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[NaN, NaN]`. At `f9030a64`: 3 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(TotalVariation1D(2f0^80), [2.0^161, 1.0], 2f0^80)
  ```
- **Classification:** contract gap (T3)
- **Tracking:** owner [#519](https://github.com/mlakolar/ProximalBase.jl/issues/519); related [#555](https://github.com/mlakolar/ProximalBase.jl/issues/555), [#554](https://github.com/mlakolar/ProximalBase.jl/issues/554)
- **Priority:** P2 — NaN or zeros from `Float32` factors
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the three functions match their widened-factor results

#### LIM-041

- **Summary:** `AtomIterate` coordinate and block steps round a wider argument to the storage type before the kernel.
- **Affected API:** `prox_coordinate!` and block steps on `AtomIterate`
- **Conditions:** `Float64` argument, `Float32` iterate
- **Expected:** T3: the argument is never rounded below the model precision
- **Actual:** the coordinate step stores `0` where the whole-array route gives `2^-27`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `Float32[0.0, 0.0]` (T3: `2^-27`). At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  y = AtomIterate(Float32, 2, [1:1, 2:2])
  prox_coordinate!(y, GroupNormL2(1f0, [1:1, 2:2], ones(Float32, 2)), 1, [1.0 + 2.0^-28], 1.0 - 2.0^-28)
  ```
- **Classification:** contract gap (T3)
- **Tracking:** owner [#513](https://github.com/mlakolar/ProximalBase.jl/issues/513); related [#511](https://github.com/mlakolar/ProximalBase.jl/issues/511)
- **Priority:** P3 — one decision near a threshold in mixed precision
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer stores `2^-27`

#### LIM-042

- **Summary:** `value_and_prox!` of group penalties with a narrower argument and a wider destination is not the value of the stored output.
- **Affected API:** `value_and_prox!` of `GroupNormL2`, `GroupMCP`
- **Conditions:** `Float32` argument, `Float64` destination
- **Expected:** O2: the value at the stored output
- **Actual:** relative gap `1.45e-8` between the returned value and `f(y)` for `GroupNormL2`, and `6.97e-9` for `GroupMCP(1, [[1, 2, 3]]; a = 2)` at the same argument
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `(2.1438608169555664, 2.1438607858227723)`. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  f = GroupNormL2(1.0, [1:3], [1.0]); y = zeros(3)
  value_and_prox!(y, f, Float32[1.1, 2.3, -0.7], 0.5), f(y)
  ```
- **Classification:** contract gap (O2, T3)
- **Tracking:** owner [#533](https://github.com/mlakolar/ProximalBase.jl/issues/533); related [#511](https://github.com/mlakolar/ProximalBase.jl/issues/511)
- **Priority:** P3 — gap at `Float32` scale in mixed precision
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the value equals `f(y)` to a few `eps(Float64)`

#### LIM-043

- **Summary:** Group penalties compute the group norm in the narrower argument type.
- **Affected API:** `prox!` of `GroupNormL2`
- **Conditions:** `Float32` argument, `Float64` destination
- **Expected:** T3: work in the model precision
- **Actual:** results differ from the call with the widened argument by `8.7e-10` (group 1) and `1.93e-9` (group 2) relative on the reproducer below (`5.4e-10` on the audit's random data)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: maximum relative difference 1.93e-9 on the reproducer below (≈ 5e-10 on the audit's random data). At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  f = GroupNormL2(0.3, [1:3, 4:6], ones(2)); x = Float32[1.1, 2.3, -0.7, 0.3, -1.9, 2.2]
  prox!(zeros(6), f, x, 0.5) .- prox!(zeros(6), f, Float64.(x), 0.5)
  ```
- **Classification:** contract gap (T3)
- **Tracking:** owner [#484](https://github.com/mlakolar/ProximalBase.jl/issues/484) (group tracker; no specific issue yet); related [#533](https://github.com/mlakolar/ProximalBase.jl/issues/533), [#511](https://github.com/mlakolar/ProximalBase.jl/issues/511)
- **Priority:** P3 — small, mixed precision only
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the two calls agree to a few `eps(Float64)`

#### LIM-044

- **Summary:** `value_and_prox!` of `Precompose` around an inner function without a standalone value returns the value in the narrower image precision.
- **Affected API:** `value_and_prox!` of `Precompose(f, a)` with, e.g., `f = Conjugate(ElasticNet(…))`
- **Conditions:** `Float32` argument and scale, `Float64` destination
- **Expected:** O2 and T3: the value at the `Float64` stored output
- **Actual:** relative gap `2e-7` to the BigFloat value at the stored output (3e-16 with `Float64` parameters)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: relative gap ≈ 2e-7. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  P = Precompose(Conjugate(ElasticNet(1f0, 0.5f0)), Float32[2, -0.5, 1.5])
  value_and_prox!(zeros(3), P, Float32[1, 2, 3], 0.5)
  ```
- **Classification:** contract gap (O2, T3)
- **Tracking:** owner [#490](https://github.com/mlakolar/ProximalBase.jl/issues/490) (group tracker; no specific issue yet); related [#533](https://github.com/mlakolar/ProximalBase.jl/issues/533)
- **Priority:** P2 — `Float32`-accuracy value where `Float64` is delivered
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the returned value matches the BigFloat value at the stored output to a few `eps(Float64)`

#### LIM-045

- **Summary:** `Float16` is not rejected by role: `Float16` arguments, destinations, directions and data are accepted and computed.
- **Affected API:** nearly every oracle (separable functions, group norms, balls, simplex, `Constrained`, `Conjugate`, wrappers, `gradient!`, `hessian_vec!`, data losses); only the LAPACK families reject
- **Conditions:** `Float16` array in any operand role
- **Expected:** T6 and decision 7: rejected before writing in 0.6; `Float16` parameters and steps accepted
- **Actual:** accepted
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: both succeed. At `f9030a64`: 76 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox!(zeros(Float32, 3), NormL1(1.0), Float16[1, -2, 3], 0.3)
  prox!(zeros(Float16, 3), NormL1(1.0), Float32[1, -2, 3], 0.3)
  ```
- **Classification:** contract gap (T6, S1, decision 7)
- **Tracking:** owner [#565](https://github.com/mlakolar/ProximalBase.jl/issues/565) (opened from this audit); related [#492](https://github.com/mlakolar/ProximalBase.jl/issues/492) (G9), [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489) (LIM-046), [#313](https://github.com/mlakolar/ProximalBase.jl/issues/313)
- **Priority:** P2 — a decided 0.6 rule is not enforced
- **Release impact:** Not yet decided (Q2)
- **Resolution criteria:** every oracle family raises `ArgumentError` before writing for a `Float16` argument, destination, direction or data, and accepts `Float16` steps and parameters

#### LIM-046

- **Summary:** LAPACK-backed functions accept `Float16`, `BigFloat` and integer arguments when the output is `Float32` or `Float64`.
- **Affected API:** `IndPSD`, `NuclearNorm`, `LogDetLoss`, `Spectral`, `SingularValue`, and the `NuclearNorm(0)`/full-rank `IndBallRank` shortcuts
- **Conditions:** argument of another element type than the output
- **Expected:** A5 and the spectral docstring: both the argument's and the output's element types are checked
- **Actual:** accepted; the argument is converted to the output's precision
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: succeeds. At `f9030a64`: 11 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox!(zeros(2, 2), IndPSD(), BigFloat[1 0; 0 1], 1.0)
  ```
- **Classification:** contract gap (A5, T6, T7)
- **Tracking:** owner [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489); related [#366](https://github.com/mlakolar/ProximalBase.jl/issues/366)
- **Priority:** P2 — the docstring promise is not kept
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer and its `Float16`/`Int` variants raise `ArgumentError` before writing

### NaN, infinities and degenerate parameters (E1–E6)

#### LIM-047

- **Summary:** `NormL2` and `GroupNormL2` proximal outputs turn a NaN entry into zeros.
- **Affected API:** `prox!` of `NormL2`, `GroupNormL2`, `Constrained(GroupNormL2, IndBallL2)` (CPU and device)
- **Conditions:** NaN entry in the vector or group
- **Expected:** E2 and decision 20: `prox(NormL2(1.0), [NaN, 4.0], 1.0)` is `[NaN, NaN]` (the contract's own example)
- **Actual:** `prox(NormL2(1.0), [NaN, 4.0], 1.0) == [0.0, 0.0]`; `GroupNormL2` with groups `1:2`, `3:4` at `[NaN, 4, 3, 4]` gives zeros in group 1 and the correct group 2; `value_and_prox!` of both returns zeros in the NaN block with a finite value (`0.0` for `NormL2`); `Constrained(GroupNormL2, IndBallL2)` at `[NaN, 0]` gives `[0.0, 0.0]`; JLArrays give the same
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[0.0, 0.0]`. At `f9030a64`: 8 cases in `reproduce-active.jl` print `PRESENT` on Julia 1.13.3 (3 of them device cases with JLArrays) and the 5 CPU cases on 1.10.12.
  ```julia
  prox(NormL2(1.0), [NaN, 4.0], 1.0)
  ```
- **Classification:** contract gap (E2, decision 20)
- **Tracking:** owner [#559](https://github.com/mlakolar/ProximalBase.jl/issues/559) (opened from this audit); related [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) (G2), [#557](https://github.com/mlakolar/ProximalBase.jl/issues/557)
- **Priority:** P1 — a NaN iterate silently becomes zeros
- **Release impact:** Proposed: blocks 0.6
- **Resolution criteria:** the reproducer is `[NaN, NaN]`; the NaN group of `GroupNormL2` is NaN and an independent group keeps its finite result; `Constrained(GroupNormL2, IndBallL2)` at `[NaN, 0]` is `[NaN, NaN]`; `value_and_prox!` returns NaN in the NaN block with value NaN

<a id="lim-048"></a>

#### LIM-048a

- **Summary:** L0 penalties and cardinality sets select a finite branch, or declare feasibility, at a NaN entry. Split from LIM-048 after the review of #558; LIM-048b and LIM-048c are the other parts.
- **Affected API:** `prox_entry`, `prox!`, `value_and_prox!` and values of `NormL0`; values, `prox!` and `value_and_prox!` of `IndBallL0`
- **Conditions:** NaN entry
- **Expected:** E2: a comparison with NaN never selects a branch; NaN in exactly the outputs and values that depend on the NaN (`NormL0` thresholds entry by entry; the top-`k` selection of `IndBallL0` couples the entries when `k < n`)
- **Actual:** `prox_entry(NormL0(1.0), NaN, 1, 1.0) == 0`; `prox(NormL0(1.0), [NaN, 1.5], 1.0) == [0, 1.5]`; `NormL0(1.0)([NaN, 1.5]) == 2.0`; `IndBallL0(1)([NaN, 0]) == 0` (false feasibility); `prox(IndBallL0(1), [NaN, 1.5], 1.0) == [0, 1.5]`, and `value_and_prox!` gives the value `0`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `0.0`; `0.0`. At `f9030a64`: 6 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox_entry(NormL0(1.0), NaN, 1, 1.0)
  IndBallL0(1)([NaN, 0.0])
  ```
- **Classification:** contract gap (E2, decision 20)
- **Tracking:** owner [#560](https://github.com/mlakolar/ProximalBase.jl/issues/560) (opened from this audit); related [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) (G2), [#561](https://github.com/mlakolar/ProximalBase.jl/issues/561) (LIM-048b), [#562](https://github.com/mlakolar/ProximalBase.jl/issues/562) (LIM-048c)
- **Priority:** P1 — NaN silently becomes a finite value or a feasible point; found in this audit, not listed in #366
- **Release impact:** Proposed: blocks 0.6 (Q2 open)
- **Resolution criteria:** `NormL0` gives NaN in the NaN entry and its value is NaN; `IndBallL0(1)` gives NaN in both outputs and the value at `[NaN, 1.5]`, and its value at `[NaN, 0]` is NaN

#### LIM-048b

- **Summary:** Scalar penalties and losses select a finite branch at a NaN entry. Split from LIM-048 after the review of #558.
- **Affected API:** `prox` of `LogPenalty`, `CheckLoss`, `HingeLoss`; values of `MCP`, `SCAD`
- **Conditions:** NaN entry
- **Expected:** E2: NaN in the NaN entry (these maps are separable, so the other entries keep their finite results); a value that sums over the entries is NaN
- **Actual:** `prox` at `[NaN, 1.5]` gives `[0.0, 1.0]` for `LogPenalty(1, 1)` and `CheckLoss(0.5)`, and `[1.0, 1.5]` for `HingeLoss([1, 1])`; `MCP(1, 3)([NaN]) == 1.5`, `SCAD(1, 3.7)([NaN]) == 2.35`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: as above. At `f9030a64`: 5 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(CheckLoss(0.5), [NaN, 1.5], 1.0)
  MCP(1.0, 3.0)([NaN])
  ```
- **Classification:** contract gap (E2, decision 20)
- **Tracking:** owner [#561](https://github.com/mlakolar/ProximalBase.jl/issues/561) (opened from this audit); related [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) (G2), [#560](https://github.com/mlakolar/ProximalBase.jl/issues/560) (LIM-048a), [#562](https://github.com/mlakolar/ProximalBase.jl/issues/562) (LIM-048c), [#519](https://github.com/mlakolar/ProximalBase.jl/issues/519), [#554](https://github.com/mlakolar/ProximalBase.jl/issues/554)
- **Priority:** P1 — NaN silently becomes a finite value or a feasible point; found in this audit, not listed in #366
- **Release impact:** Proposed: blocks 0.6 (Q2 open)
- **Resolution criteria:** each call gives NaN in the NaN entry, the finite result elsewhere, and a NaN value

#### LIM-048c

- **Summary:** Ordered, fused and isotonic proximal maps return finite outputs that depend on a NaN entry. Split from LIM-048 after the review of #558.
- **Affected API:** `prox!`, `value_and_prox!` of `SLOPE`, `NormL1Fused`, `IndIsotonic`
- **Conditions:** NaN entry
- **Expected:** E2: every output that depends on the NaN is NaN; here all of them do (the rank of each entry selects its `SLOPE` weight, the total-variation term couples all entries of `NormL1Fused`, pooling merges neighbours transitively in `IndIsotonic`)
- **Actual:** at `[1, NaN, 0.5]`: `SLOPE([3, 2, 1])` gives `[0, NaN, 0]`, `NormL1Fused(1, 1)` gives `[2.0, NaN, -0.5]`, `IndIsotonic()` gives `[1, NaN, 0.5]` (the finite entries 1 and 3 violate the order)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[2.0, NaN, -0.5]`. At `f9030a64`: 3 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(NormL1Fused(1.0, 1.0), [1.0, NaN, 0.5], 1.0)
  ```
- **Classification:** contract gap (E2, decision 20)
- **Tracking:** owner [#562](https://github.com/mlakolar/ProximalBase.jl/issues/562) (opened from this audit); related [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) (G2), [#560](https://github.com/mlakolar/ProximalBase.jl/issues/560) (LIM-048a), [#561](https://github.com/mlakolar/ProximalBase.jl/issues/561) (LIM-048b), [#554](https://github.com/mlakolar/ProximalBase.jl/issues/554), [#361](https://github.com/mlakolar/ProximalBase.jl/issues/361)
- **Priority:** P1 — NaN silently becomes a finite value or a feasible point; found in this audit, not listed in #366
- **Release impact:** Proposed: blocks 0.6 (Q2 open)
- **Resolution criteria:** the three calls return NaN in every entry; `test/threshold_factor_type.jl:144` is updated

#### LIM-049

- **Summary:** Standalone values at a NaN argument are `+Inf` for indicators and several functions, instead of NaN.
- **Affected API:** values of indicators and indicator conjugates, `NegEntropy`, `NegLog`, `NormL1Nonneg`, the spectral functions, `LogDetLoss`, `IndIsotonic`
- **Conditions:** NaN entry
- **Expected:** E2 and decision 20: a failed membership test must not turn NaN into `+Inf`; the standalone value is NaN
- **Actual:** `Inf` for each of 22 functions: `IndBox`, `IndNonnegative`, `IndBallL2`, `IndBallL1`, `IndSimplex`, `IndBallLinf`, `IndBinary`, `IndSOC`, `IndHalfspace`, `IndHyperplane`, `IndHyperslab`, `Conjugate(NormL1)`, `Conjugate(SqrHingeLoss)`, `NegEntropy`, `NegLog`, `NormL1Nonneg`, `IndIsotonic`, `IndPSD`, `NuclearNorm`, `SingularValue`, `Spectral`, `LogDetLoss` (`Conjugate(IndBox)` already gives NaN)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `(Inf, Inf)`. At `f9030a64`: 22 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  IndBox(-1.0, 1.0)([NaN]), NegEntropy()([NaN])
  ```
- **Classification:** contract gap (E2, decision 20)
- **Tracking:** owner [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) (group tracker; no specific issue yet); related [#349](https://github.com/mlakolar/ProximalBase.jl/issues/349), [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489), [#366](https://github.com/mlakolar/ProximalBase.jl/issues/366)
- **Priority:** P2 — wrong value class at NaN, widespread; several tests pin `Inf`
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** these values are NaN at a NaN argument; `test/spectral_nonfinite_values.jl` and `test/nonfinite_inputs.jl` are updated

#### LIM-050

- **Summary:** `value_and_prox!` of indicators returns `0` although the stored output contains NaN.
- **Affected API:** `value_and_prox!` of `IndBox`, `IndNonnegative`, `IndBallLinf`, `IndIsotonic`, `Conjugate(NormL1/NormL2/NuclearNorm)`, block sums of boxes, array-step `IndBallL1`/`IndSimplex`; `Conjugate(SingularValue(…))` raises instead
- **Conditions:** NaN (or an `Inf` that the projection maps to NaN) in the argument
- **Expected:** O2 and E2: no false zero; the zero-value shortcut does not apply to a NaN projection; a prescribed NaN is a result, not a failure
- **Actual:** value `0.0` with `y = [NaN, 0.0]`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `0.0`. At `f9030a64`: 14 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  y = zeros(2); value_and_prox!(y, IndBox(-1.0, 1.0), [NaN, 0.0], 1.0)
  ```
- **Classification:** contract gap (O2, E2)
- **Tracking:** owner [#349](https://github.com/mlakolar/ProximalBase.jl/issues/349); related [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485), [#350](https://github.com/mlakolar/ProximalBase.jl/issues/350)
- **Priority:** P2 — false feasibility value at a NaN output
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the value is NaN whenever the stored output depends on a NaN; `Conjugate(SingularValue(…))` returns a NaN block

#### LIM-051

- **Summary:** `value_and_prox!` of `Precompose` around an indicator returns `0` at a stored NaN (scalar scale, array scale, and mixed precision for `IndBallL2`, `IndBallL1` and `IndPSD`).
- **Affected API:** `value_and_prox!` of `Precompose(indicator, a)` (scalar, array scale, mixed precision)
- **Conditions:** NaN argument; a `Float32` output from a `Float64` argument matters only for the ball and spectral indicators (`Precompose(IndBox, a)` fails in plain `Float64`); at an `Inf` argument or a finite point outside the set the mixed-precision calls are correct
- **Expected:** E2 and O2: the value at the stored output, NaN when the stored output depends on a NaN; no false zero
- **Actual:** value `0.0` with `y = [NaN, NaN]`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `0.0`. At `f9030a64`: 5 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  value_and_prox!(fill(7f0, 2), Precompose(IndBallL2(1.0), 2.0), [NaN, 0.0], 1.0)
  ```
- **Classification:** contract gap (O2, E5)
- **Tracking:** owner [#350](https://github.com/mlakolar/ProximalBase.jl/issues/350); related [#349](https://github.com/mlakolar/ProximalBase.jl/issues/349)
- **Priority:** P2 — false zero value
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the value is NaN (or `Inf`) with the class of the standalone call, in every precision mix

#### LIM-052

- **Summary:** A coordinate fixed by an infinite weight is NaN when the argument there is NaN or infinite.
- **Affected API:** `prox!` of `NormL1`, `ElasticNet`, `CappedL1`, `Berhu`, `LogPenalty`, `NormLq` with infinite weights (not `EpsInsensitiveLoss`: an infinite weight confines its coordinate to the tube `[-ε, ε]`, so that output depends on the argument and NaN is E2's result)
- **Conditions:** argument entry NaN or `±Inf` at the fixed coordinate
- **Expected:** E2 and decision 48: a structurally fixed output does not depend on the exceptional input; it stays `0`
- **Actual:** `prox(NormL1(1, [Inf, 1]), [NaN, 2], 1) == [NaN, 1]`; `test/infinite_weights.jl:58` pins the NaN
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[NaN, 1.0]`. At `f9030a64`: 14 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(NormL1(1.0, [Inf, 1.0]), [NaN, 2.0], 1.0)
  ```
- **Classification:** contract gap (E2, decision 48)
- **Tracking:** owner [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) (group tracker; no specific issue yet); related [#469](https://github.com/mlakolar/ProximalBase.jl/issues/469)
- **Priority:** P2 — a fixed coordinate becomes NaN
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer gives `[0, 1]` for every listed family, and also at `±Inf` for `NormL1`, `ElasticNet`, `CappedL1` and `Berhu` (`LogPenalty` and `NormLq` already give `0` there)

#### LIM-053

- **Summary:** A zero level does not give the zero function at non-finite arguments.
- **Affected API:** value of `NuclearNorm(0)` and full-rank `IndBallRank`; `gradient!` and `γ = Inf` prox of `SqrNormL2(0)`
- **Conditions:** NaN or `±Inf` entry; `γ = Inf`
- **Expected:** E4 and decision 22: value 0, gradient 0, proximal step that copies the argument. For a full-rank `IndBallRank` the value is taken as `0` (the set is the whole space); reading it as an indicator under E2/E3 (NaN at NaN, `+Inf` at `Inf`) would also be defensible, and the reproducer would then print `CHANGED` for its two cases (decision needed)
- **Actual:** `NuclearNorm(0)([NaN 0; 0 1]) == Inf`; `gradient!(g, SqrNormL2(0), [NaN, 4]) == [NaN, 0]`; `prox(SqrNormL2(0), [1, 2], Inf)` is `[NaN, NaN]`; also at `Inf` arguments and in the fused values
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `(Inf, [NaN, 0.0])`. At `f9030a64`: 9 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  NuclearNorm(0.0)([NaN 0; 0 1]), gradient!(zeros(2), SqrNormL2(0.0), [NaN, 4.0])
  ```
- **Classification:** contract gap (E4, decision 22)
- **Tracking:** owner [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) (group tracker; no specific issue yet); related [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489), [#366](https://github.com/mlakolar/ProximalBase.jl/issues/366)
- **Priority:** P2 — zero-level functions misbehave at exceptional inputs
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** value 0, gradient 0 and the identity step for these calls

#### LIM-054

- **Summary:** The `MoreauEnvelope` value at an infinite entry is NaN even where the limit is `+Inf`.
- **Affected API:** value of `MoreauEnvelope(f, μ)` (e.g. `f = NormL1`, `NuclearNorm`)
- **Conditions:** `±Inf` entry
- **Expected:** E3: the formula or limit where determined (the envelope of `NormL1` is the Huber function, `+Inf` at `Inf`)
- **Actual:** `NaN` (from `Inf − Inf` in the formula), for `MoreauEnvelope(NormL1(1.0), 1.0)` at `[Inf]` and `MoreauEnvelope(NuclearNorm(1.0), 1.0)` at `[Inf 0; 0 1]`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `NaN`. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  MoreauEnvelope(NormL1(1.0), 1.0)([Inf])
  ```
- **Classification:** contract gap (E3)
- **Tracking:** owner [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) (group tracker; no specific issue yet); related [#490](https://github.com/mlakolar/ProximalBase.jl/issues/490)
- **Priority:** P3 — narrow
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer returns `Inf`

#### LIM-055

- **Summary:** The determined limits of decision 48 are not implemented for the remaining constrained families and the `NormLinf` conjugates.
- **Affected API:** `prox!` of `Constrained(IndBallL0, IndBallL2/IndSphereL2)`, `Constrained(ZeroFunction, IndBallL2)`, `Constrained(GroupNormL2, IndBallL2/IndSphereL2)` (PR #557 open), `Constrained(Scaled(NormL1), IndBallL2)`; `Conjugate(NormLinf)`, `Conjugate(GroupNormLinf)`
- **Conditions:** one `±Inf` entry
- **Expected:** E3 and decision 48: the path-independent limit, NaN only in the dependent entries
- **Actual:** all-NaN blocks (`[NaN, NaN]` for `Constrained(IndBallL0(1), IndSphereL2(1))` at `[Inf, 0]`, limit `[1, 0]`); `Conjugate(NormLinf)` gives the partly NaN `[NaN, 0, 0]` (limit `[1, 0, 0]`) and `Conjugate(GroupNormLinf(1, [1:2, 3:3]))` gives `[NaN, 0, 0.2]` (limit `[1, 0, 0.2]`); the six constrained families listed all give `[NaN, NaN]` at `[Inf, 0]`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[NaN, NaN]`; `[NaN, 0.0, 0.0]`. At `f9030a64`: 8 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(Constrained(IndBallL0(1), IndSphereL2(1.0)), [Inf, 0.0], 1.0)
  prox(Conjugate(NormLinf(1.0)), [Inf, 0.5, 0.2], 1.0)
  ```
- **Classification:** contract gap (E3, decision 48)
- **Tracking:** owner [#469](https://github.com/mlakolar/ProximalBase.jl/issues/469); related [#557](https://github.com/mlakolar/ProximalBase.jl/issues/557), [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485), [#372](https://github.com/mlakolar/ProximalBase.jl/issues/372)
- **Priority:** P2 — decided limits not yet implemented (staged work)
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** each listed family returns its limit at one infinite entry

#### LIM-056

- **Summary:** At the admitted endpoint `γ = Inf`, a NaN argument entry gives NaN where the map is constant.
- **Affected API:** `prox!` of `NormL1` and the separable families at `γ = Inf`
- **Conditions:** NaN entry, `γ = Inf`
- **Expected:** E2 and decision 48: `prox(NormL1(1.0), [NaN], Inf)` is `[0]`
- **Actual:** `[NaN, 0.0]` (numerics.md notes that current implementations may return NaN)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[NaN, 0.0]`. At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(NormL1(1.0), [NaN, 1.0], Inf)
  ```
- **Classification:** contract gap (E2, decision 48)
- **Tracking:** owner [#469](https://github.com/mlakolar/ProximalBase.jl/issues/469); related [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311), [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485)
- **Priority:** P2 — named in the contract as not yet conforming
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer returns `[0, 0]`

#### LIM-057

- **Summary:** The value of `Scaled(indicator, Inf)` is NaN.
- **Affected API:** value and `value_and_prox!` of `Scaled(f, Inf)` for an indicator `f`
- **Conditions:** any argument inside the set; any step
- **Expected:** O2 and E4: `0` inside the set, `Inf` outside; the joint value `0` at the projection
- **Actual:** `NaN` (`Inf * 0`); the projection itself is correct
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `NaN`. At `f9030a64`: 3 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  Scaled(IndBallL2(1.0), Inf)([0.3, 0.4])
  ```
- **Classification:** contract gap (O2, E4)
- **Tracking:** owner [#311](https://github.com/mlakolar/ProximalBase.jl/issues/311); related [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485), [#522](https://github.com/mlakolar/ProximalBase.jl/issues/522)
- **Priority:** P2 — NaN value from a constructed function
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the value is `0` inside the set and the joint value is `0`

#### LIM-058

- **Summary:** `Scaled(f, 0)` is rejected for many functions whose zero scaling has a domain-preserving representation.
- **Affected API:** `Scaled(f, 0)` for `HingeLoss`, `SqrHingeLoss`, `CheckLoss`, `EpsInsensitiveLoss`, `LogisticLoss`, `TotalVariation1D`, `FusedLasso`, `SLOPE`, `NormL1Fused`, `NuclearNorm`, `MoreauEnvelope`, `Translate` (total domain: the zero function is expected), and `KLDivergence`, `NegEntropy` (domain `x ≥ 0`: the indicator of the domain is expected). Not `BernoulliLoss`: its domain is open at 0, so no closest point exists and E4 allows the rejection, as for `NegLog` (LIM-087)
- **Conditions:** construction with scale 0
- **Expected:** E4: a function without domain constraints becomes the zero function; rejection only where no domain-preserving representation exists
- **Actual:** `ArgumentError: zero scaling of … has no supported domain-preserving representation`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `ArgumentError`. At `f9030a64`: 14 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  Scaled(HingeLoss(ones(3)), 0.0)
  ```
- **Classification:** contract gap (E4, S4)
- **Tracking:** owner [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) (group tracker; no specific issue yet); related [#333](https://github.com/mlakolar/ProximalBase.jl/issues/333)
- **Priority:** P3 — a degenerate but legal parameter is rejected
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** `Scaled(f, 0)` simplifies to the zero function for each listed total-domain function and keeps the domain `x ≥ 0` for `KLDivergence` and `NegEntropy`

#### LIM-059

- **Summary:** Fused spectral values disagree with the value of the stored output.
- **Affected API:** `value_and_prox!` of `SingularValue(NormL0)`, `Spectral(NormL0)`, `IndBallRank` (subnormal scale)
- **Conditions:** kept singular values below the `√eps·σ₁` mask; rank-one input at subnormal scale stored in `Float32`
- **Expected:** A5 and O2: the value describes the stored matrix (O3 names only `IndIntersection` and `LogDetLoss`)
- **Actual:** `SingularValue(NormL0(1.0))` at `Diagonal([2^60, 2, 0])` returns `2` while `f(Y) == 1`; `Spectral(NormL0(1.0))` gives the same `2` against `1`; `IndBallRank(1)` with a `Float32` output returns `0` while `f(Y) == Inf` (a `Float64` output is consistent, and `SingularValue`/`Spectral(NormL0)` are consistent at that subnormal input)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `(2.0, 1.0)`. At `f9030a64`: 3 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  Y = zeros(3, 3); f = SingularValue(NormL0(1.0))
  value_and_prox!(Y, f, Matrix(Diagonal([2.0^60, 2.0, 0.0])), 1.0), f(Y)
  ```
- **Classification:** contract gap (O2, A5)
- **Tracking:** owner [#564](https://github.com/mlakolar/ProximalBase.jl/issues/564) (opened from this audit); related [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489) (G6), [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283), [#307](https://github.com/mlakolar/ProximalBase.jl/issues/307)
- **Priority:** P2 — fused value disagrees with the stored output
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the fused value equals `f(Y)` for these functions (conformance L5 rows pass)

### Devices (G1–G6)

#### LIM-060

- **Summary:** Device kernels of `IndBallL1`, `IndSimplex` and `Constrained(NormL1, IndBallL2)` do not classify NaN and `Inf`, and accept negative and NaN steps.
- **Affected API:** `prox!`, `value_and_prox!` on GPUArrays (checked with JLArrays)
- **Conditions:** NaN or `±Inf` argument entries; `γ < 0` or NaN
- **Expected:** G1: the same scope of NaN and `Inf` as the CPU (E2/E3, decision 48) and the same step rejection (P1)
- **Actual:** `IndBallL1` at `[Inf, 0.5, 0.2]` gives `[NaN, 0, 0]` (CPU `[1, 0, 0]`); at a NaN entry a partly NaN block; `value_and_prox!` returns `0` where the CPU returns NaN; `prox!` writes the result for negative and NaN steps (`value_and_prox!` rejects them, as on the CPU); `Constrained(NormL1, IndBallL2)` gives `[NaN, NaN, NaN]` at one `Inf` (CPU `[1, 0, 0]`) and `[NaN, 0, 0]` at a NaN (CPU all NaN)
- **Evidence:** at `920ebece` with JLArrays on Julia 1.13.3: `[NaN, 0.0, 0.0]`. At `f9030a64`: 14 cases in `reproduce-active.jl` print `PRESENT` with JLArrays on Julia 1.13.3 (JLArrays does not load on 1.10).
  ```julia
  prox(IndBallL1(1.0), JLArray([Inf, 0.5, 0.2]), 1.0)
  ```
- **Classification:** contract gap (G1, E2, E3, P1)
- **Tracking:** owner [#494](https://github.com/mlakolar/ProximalBase.jl/issues/494); related [#469](https://github.com/mlakolar/ProximalBase.jl/issues/469), [#532](https://github.com/mlakolar/ProximalBase.jl/issues/532), [#545](https://github.com/mlakolar/ProximalBase.jl/issues/545)
- **Priority:** P2 — device results differ from the CPU contract (left on the old algorithm in #529)
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** JLArray results equal the CPU results for NaN, one and two infinite entries, and, for `prox!`, invalid steps

#### LIM-061

- **Summary:** `Conjugate(NormLinf)` and `Conjugate(GroupNormLinf)` on device arrays cancel through Moreau's identity.
- **Affected API:** `prox!` of `Conjugate(NormLinf)`, `Conjugate(GroupNormLinf)` on GPUArrays
- **Conditions:** `max|x|` far above the radius
- **Expected:** G1 and R2: the CPU's direct ℓ1-ball projection result
- **Actual:** `[0, 0, 0]` at `[1e300, 0.25, -0.5]` (CPU `[1, 0, -0]`); `test/gpu_suite.jl:1746` pins the cancelling result
- **Evidence:** at `920ebece` with JLArrays on Julia 1.13.3: `[0.0, 0.0, 0.0]`. At `f9030a64`: 2 cases in `reproduce-active.jl` print `PRESENT` with JLArrays on Julia 1.13.3 (JLArrays does not load on 1.10).
  ```julia
  prox(Conjugate(NormLinf()), JLArray([1e300, 0.25, -0.5]), 1.0)
  ```
- **Classification:** contract gap (G1, R2)
- **Tracking:** owner [#372](https://github.com/mlakolar/ProximalBase.jl/issues/372); related [#494](https://github.com/mlakolar/ProximalBase.jl/issues/494), [#370](https://github.com/mlakolar/ProximalBase.jl/issues/370)
- **Priority:** P2 — silently wrong device result
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the device result equals the CPU result within `4n·eps·r`

#### LIM-062

- **Summary:** Device kernels narrow a host level to the element type before applying the weight.
- **Affected API:** device kernels of `ElasticNet` and its conjugate (and other weighted kernels per #322)
- **Conditions:** level not representable in the element type, e.g. `1e40` with `Float32` data
- **Expected:** G5 and decision 34: risky scalar parameters are decided on the host, not narrowed
- **Actual:** `Conjugate(ElasticNet(1e40, 1, Float32[1e-30]))` at `Float32[2e10]`: device `2.0f10`, CPU `1.5f10` (exact `1.5e10`); `Conjugate(ElasticNet(big"1e400", 1, [1e-300]))` at `[2e100]`: device `2.0e100`, CPU `1.5e100`
- **Evidence:** at `920ebece` with JLArrays on Julia 1.13.3: `Float32[1.5f10]` vs `Float32[2.0f10]`. At `f9030a64`: 2 cases in `reproduce-active.jl` print `PRESENT` with JLArrays on Julia 1.13.3 (JLArrays does not load on 1.10).
  ```julia
  f = ElasticNet(1e40, 1.0, Float32[1e-30])
  prox(Conjugate(f), Float32[2e10], 1f0), prox(Conjugate(adapt(JLArray, f)), JLArray(Float32[2e10]), 1f0)
  ```
- **Classification:** contract gap (G5, G1, R3)
- **Tracking:** owner [#322](https://github.com/mlakolar/ProximalBase.jl/issues/322); related [#494](https://github.com/mlakolar/ProximalBase.jl/issues/494)
- **Priority:** P2 — silently wrong device result for out-of-range parameters
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the #322 cases on JLArrays equal the CPU

#### LIM-063

- **Summary:** Several wrappers index scalars or use CPU buffers and fail on device arrays.
- **Affected API:** `MoreauEnvelope` (value, prox), `Translate` (value), `SqrDistL2`, `DistL2`, scalar and array-scale `Precompose`
- **Conditions:** device arrays with scalar indexing disallowed
- **Expected:** numerics.md Support table: wrappers run on the backends of their inner function (G1)
- **Actual:** the scalar-indexing `ErrorException`; outputs unchanged
- **Evidence:** at `920ebece` with JLArrays on Julia 1.13.3: `ErrorException: Scalar indexing is disallowed`. At `f9030a64`: 16 cases in `reproduce-active.jl` print `PRESENT` with JLArrays on Julia 1.13.3 (JLArrays does not load on 1.10).
  ```julia
  MoreauEnvelope(NormL1(1.0), 0.5)(JLArray([1.0, -2.0, 0.3]))
  ```
- **Classification:** contract gap (G1, S4)
- **Tracking:** owner [#327](https://github.com/mlakolar/ProximalBase.jl/issues/327); related [#494](https://github.com/mlakolar/ProximalBase.jl/issues/494)
- **Priority:** P3 — loud failure of a documented backend
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** values, `prox!` and `value_and_prox!` of these wrappers on JLArrays equal the CPU

### Wrappers, conjugates and other oracles

#### LIM-064

- **Summary:** `value_and_prox!` of `Constrained(f, IndBox)` raises for an inner function without a standalone value, although the leaf supports it.
- **Affected API:** `value_and_prox!` of `Constrained(Conjugate(ElasticNet(…)), IndBox)`
- **Conditions:** inner function with a proximal value but no standalone value
- **Expected:** O2: the joint value from the inner proximal computation, as the leaf returns
- **Actual:** `ArgumentError: the value of the conjugate of ElasticNet is not implemented`; output unchanged
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `ArgumentError`. At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  value_and_prox!(zeros(2), Constrained(Conjugate(ElasticNet(1.0, 1.0)), IndBox(-1.0, 1.0)), [0.5, 2.0], 1.0)
  ```
- **Classification:** contract gap (O2)
- **Tracking:** owner [#316](https://github.com/mlakolar/ProximalBase.jl/issues/316); related [#490](https://github.com/mlakolar/ProximalBase.jl/issues/490)
- **Priority:** P2 — wrapper fails where the leaf succeeds
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the call returns the joint value

#### LIM-065

- **Summary:** Wrappers apply different shape rules to per-entry parameter arrays.
- **Affected API:** `LinearFunction`, `Tilt`, `Translate`, centred `SqrNormL2`, weighted `NormL1`
- **Conditions:** parameter array of equal length but another shape
- **Expected:** one documented shape rule for per-entry parameters (E1 order of checks)
- **Actual:** `Tilt` and `LinearFunction` accept a length-4 vector for a 2×2 argument; `Translate` raises `DimensionMismatch`; centred `SqrNormL2` raises in its value and in `prox!`, while the raw kernel `prox_entry` (which checks no parameters by design) does not
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `DimensionMismatch` vs `39.0`. At `f9030a64`: 3 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  Translate(NormL1(1.0), [1.0, 2, 3, 4])([1.0 2; 3 4]), Tilt(NormL1(1.0), [1.0, 2, 3, 4])([1.0 2; 3 4])
  ```
- **Classification:** contract gap (E1)
- **Tracking:** owner [#312](https://github.com/mlakolar/ProximalBase.jl/issues/312); related [#294](https://github.com/mlakolar/ProximalBase.jl/issues/294)
- **Priority:** P3 — inconsistent validation, no wrong numerics
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** one shape rule applied by all five in values, gradients and `prox!`; `prox_entry` stays unchecked

#### LIM-066

- **Summary:** `restrict` of `IndBinary` requires every bound pair to contain zero, even when only the kept coordinates matter.
- **Affected API:** `restrict(IndBinary(l, u), S)`, `restrict(Translate(IndBinary, b), S)`
- **Conditions:** a dropped coordinate whose pair contains zero (or `-b`), a kept one that does not
- **Expected:** O5 and the `restrict` docstring: only the dropped terms must be finite at the embedding point
- **Actual:** `ArgumentError: restrict requires zero-preserving binary bounds`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `ArgumentError`. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  restrict(IndBinary([0.0, 1.0, 0.0], [1.0, 2.0, 1.0]), [1, 2])
  restrict(Translate(IndBinary(-1.0, 1.0), [1.0, 1.0, 0.0]), [3])
  ```
- **Classification:** contract gap (O5)
- **Tracking:** owner [#264](https://github.com/mlakolar/ProximalBase.jl/issues/264); related [#296](https://github.com/mlakolar/ProximalBase.jl/issues/296)
- **Priority:** P3 — narrow
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer restricts; it still throws when a dropped pair excludes the embedding point

#### LIM-067

- **Summary:** `Conjugate(SingularValue(f))` into a `SymmetricSparseIterate` raises "must be symmetric" for an exactly symmetric argument.
- **Affected API:** `prox!` of `Conjugate(SingularValue(f))`, `Conjugate(NuclearNorm)` into `SymmetricSparseIterate`
- **Conditions:** exactly symmetric argument
- **Expected:** A1/A5 and the symmetric-matrix rule: a spectral map of a symmetric argument writes a symmetric result
- **Actual:** `ArgumentError: the value written into a SymmetricSparseIterate must be symmetric`, for `Conjugate(SingularValue(NormL1(1.0)))` and `Conjugate(NuclearNorm(1.0))`; the computed result is asymmetric by one ulp (`1.1e-16`)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `ArgumentError`. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox!(SymmetricSparseIterate(2), Conjugate(SingularValue(NormL1(1.0))), [2.0 0.5; 0.5 1.0], 1.0)
  ```
- **Classification:** contract gap (A5, M2)
- **Tracking:** owner [#407](https://github.com/mlakolar/ProximalBase.jl/issues/407); related [#142](https://github.com/mlakolar/ProximalBase.jl/issues/142)
- **Priority:** P2 — a valid supported call raises
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the call writes the symmetric result

#### LIM-068

- **Summary:** Default algorithmic tolerances are fixed numbers independent of the precision.
- **Affected API:** `IndIntersection` (`tol = 1e-10`, also for `Float32` sets), `_largest_eigenvalue`/`opnorm2` (`rtol = 1e-10`), `Testing.test_kkt` (an undocumented keyword `tol = 1e-8` in the test extension, `ext/ProximalBaseTestExt.jl`)
- **Conditions:** any element type
- **Expected:** A4: no default tolerance is a fixed number independent of the precision
- **Actual:** the defaults are literal constants for every element type
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `1.0e-10` (`Float64`). At `f9030a64`: 3 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  IndIntersection(IndBox(0f0, 1f0), IndBallL2(1f0)).tol
  ```
- **Classification:** contract gap (A4)
- **Tracking:** owner [#396](https://github.com/mlakolar/ProximalBase.jl/issues/396); related [#493](https://github.com/mlakolar/ProximalBase.jl/issues/493), [#307](https://github.com/mlakolar/ProximalBase.jl/issues/307)
- **Priority:** P3 — defaults on approximate or estimated quantities
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the defaults derive from the working precision (after #396 agrees on a discriminating observation)

#### LIM-069

- **Summary:** `Conjugate` on the Moreau route raises or misreports at valid extreme steps where the result is representable.
- **Affected API:** `prox!`, `value_and_prox!`, `prox_entry` and array-step `prox!` of `Conjugate(f)` for `f` without a step-free construction (`IndHalfspace`, `IndHyperplane`, `IndHyperslab`, affine sets, `IndBallL1` and `IndSimplex` with two or more entries, `SqrNormL2`, `ElasticNet`, `SqrHingeLoss`, mixed `SeparableSum`, spectral wrappers)
- **Conditions:** `γ = 1e-310`; `γ = 1e300` (an error in `Float32` calls, a wrong value for `Conjugate(SqrNormL2)` in `Float64`); `x/γ` overflowing; a `Float64` step that does not fit a `Float32` call, array steps
- **Expected:** R1–R3 and P1: a representable result is returned for every valid step (the stopgap, if kept, raises `PrecisionError`, as O6.3 requires; today's `ArgumentError` is the wrong class)
- **Actual:** `ArgumentError: the proximal point of the conjugate is not finite at a finite input`; `prox_entry(Conjugate(2 * IndBox(0, Inf)), 3.0, 1, 1e-310) == -Inf` (exact 0); array steps `[1e-310, 1e-310]` raise the same `ArgumentError` (its message prints the step as `2-element Vector{Float64}`); `prox_entry(Conjugate(SqrNormL2()), 3.0, 1, 1e-310)` is NaN (exact 3.0); `prox(Conjugate(SqrNormL2()), [3.0], 1e300)` returns `[4.4e-16]` (exact `3e-300`). `IndBallL1`/`IndSimplex` at `[0, 1]` pass only because #529 maps one infinite entry to its limit; at `[0.5, 1]` they still raise
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `ArgumentError` three times. At `f9030a64`: 19 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(Conjugate(IndHalfspace([1.0, 1.0], 1.0)), [0.0, 1.0], 1e-310)
  prox(Conjugate(SqrNormL2()), [1e300], 1e-10)
  prox(Conjugate(IndBallL1(1.0)), [0.5, 1.0], 1e-310)
  ```
- **Classification:** contract gap (R1, R2, R3, P1, O6.3)
- **Tracking:** owner [#255](https://github.com/mlakolar/ProximalBase.jl/issues/255); related [#376](https://github.com/mlakolar/ProximalBase.jl/issues/376), [#377](https://github.com/mlakolar/ProximalBase.jl/issues/377), [#216](https://github.com/mlakolar/ProximalBase.jl/issues/216), [#543](https://github.com/mlakolar/ProximalBase.jl/issues/543)
- **Priority:** P2 — valid steps fail or give wrong values
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** each reproducer returns the exact projection onto `γC` (or its limit); `prox_entry` and array steps agree with the scalar route

#### LIM-070

- **Summary:** `Conjugate` on the Moreau route loses results to cancellation at large arguments.
- **Affected API:** `prox!` of `Conjugate(SLOPE)`, `Conjugate(TotalVariation1D)` and other families on the Moreau route
- **Conditions:** scalar step, `|x|` far above the result
- **Expected:** R2: closed-form projections onto the dual set keep the accuracy of the ordinary computation
- **Actual:** `prox(Conjugate(SLOPE([1, .5, .25])), [1e300, .25, -.5], 1)` is `[0, .25, -.5]` (exact `[1, .25, -.5]`); `Conjugate(TotalVariation1D(1.0))` at the same point gives `[0, -0.125, -0.875]` (exact `[1, -0.125, -0.875]`), which violates its dual-ball constraint
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[0.0, 0.25, -0.5]`. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  prox(Conjugate(SLOPE([1.0, 0.5, 0.25])), [1e300, 0.25, -0.5], 1.0)
  ```
- **Classification:** contract gap (R2, A1)
- **Tracking:** owner [#255](https://github.com/mlakolar/ProximalBase.jl/issues/255); related [#256](https://github.com/mlakolar/ProximalBase.jl/issues/256), [#328](https://github.com/mlakolar/ProximalBase.jl/issues/328), [#370](https://github.com/mlakolar/ProximalBase.jl/issues/370)
- **Priority:** P2 — silently wrong projection at large inputs
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the reproducer matches the BigFloat dual projection; the `TotalVariation1D` result sums to 0

### Iterates (M1–M2 and the approved 0.6 iterate design)

#### LIM-071

- **Summary:** Strong zeros (FD-9) are not implemented: a stored zero coefficient meets `Inf` or NaN as NaN, so products depend on storage and argument order.
- **Affected API:** `dot`, `mul!`, `*` with `SparseIterate`, `AtomIterate` and `matrixview(::SymmetricSparseIterate)`
- **Conditions:** an `Inf` or NaN entry of the other operand against a stored explicit zero (or an active zero group)
- **Expected:** FD-9/FN-1 (signed off for 0.6, amends M1): a zero coefficient, stored or not, contributes nothing
- **Actual:** `dot(a, s)` changes from `2.0` to NaN when coordinate 3 is activated with value 0; `dot(a, b) != dot(b, a)` with a stored NaN; `test/matrixview.jl:252` is `@test_broken`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `2.0`, then `NaN`. At `f9030a64`: 5 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  s = SparseIterate(Float64, 3); s[2] = 1.0; a = [1.0, 2.0, Inf]
  dot(a, s); activate!(s, 3); dot(a, s)
  # A * s with A = [1 Inf; 1 1; 1 1] and a stored zero: [NaN, 1, 1], and [1, 1, 1] after dropzeros!
  ```
- **Classification:** contract gap (M1 (FD-9 amendment))
- **Tracking:** owner [#428](https://github.com/mlakolar/ProximalBase.jl/issues/428); related [#456](https://github.com/mlakolar/ProximalBase.jl/issues/456), [#391](https://github.com/mlakolar/ProximalBase.jl/issues/391), [#341](https://github.com/mlakolar/ProximalBase.jl/issues/341)
- **Priority:** P2 — result class depends on storage, exceptional operands only
- **Release impact:** Not yet decided (Q3)
- **Resolution criteria:** products and dots are equal before and after `dropzeros!` and independent of argument order (FT12 rows)

#### LIM-072

- **Summary:** Value oracles still read every entry of a sparse iterate.
- **Affected API:** values of `NormL2` (overflow recovery), `IndSimplex`, `IndBallL0` (symmetric), `LinearFunction` with a matrix coefficient, on sparse iterates
- **Conditions:** sparse iterates; any values
- **Expected:** M1 with decision 47: a sparse value may skip unstored entries (O(nnz) cost)
- **Actual:** O(p²) zeros are materialised (values are correct); four `@test_broken` rows in `test/sparse_value_cost.jl` (lines 162, 165, 169, 173; six broken results, since lines 162 and 165 run for both iterate kinds)
- **Evidence:** the four `@test_broken` rows are still broken in the full test run at `920ebece` and in `test/sparse_value_cost.jl` at `f9030a64` (1961 pass, 6 broken, on Julia 1.13.3 and 1.10.12); the script lists the entry as `SKIPPED` with that pointer.
  ```julia
  # see test/sparse_value_cost.jl, testset "Known dense paths"
  ```
- **Classification:** contract gap (M1 (cost))
- **Tracking:** owner [#341](https://github.com/mlakolar/ProximalBase.jl/issues/341); related [#142](https://github.com/mlakolar/ProximalBase.jl/issues/142)
- **Priority:** P3 — cost only
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the four `@test_broken` rows pass

#### LIM-073

- **Summary:** Scaling a `SymmetricSparseIterate` by `Inf`, NaN or a zero divisor depends on which coordinates are stored.
- **Affected API:** `rmul!`, `lmul!`, `rdiv!`, `ldiv!` of `SymmetricSparseIterate`
- **Conditions:** a scalar that maps zero to a nonzero value
- **Expected:** FN-9 / FW-3: scaling equals the broadcast `X .*= s` (implemented for `SparseIterate` and `AtomIterate` only)
- **Actual:** `rmul!(X, Inf)` leaves unstored positions 0; `X .*= Inf` makes them NaN
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `false`. At `f9030a64`: 4 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  X = SymmetricSparseIterate(Float64, 3); X[1, 1] = 1.0; X[2, 1] = 2.0
  Y = copy(X); rmul!(X, Inf); Y .*= Inf; Matrix(X) == Matrix(Y)
  ```
- **Classification:** contract gap (M1, FN-9)
- **Tracking:** owner [#391](https://github.com/mlakolar/ProximalBase.jl/issues/391); related [#427](https://github.com/mlakolar/ProximalBase.jl/issues/427), [#428](https://github.com/mlakolar/ProximalBase.jl/issues/428)
- **Priority:** P2 — wrong class for exceptional scalars
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** `rmul!(X, s)` equals `X .*= s` in values and working set

#### LIM-074

- **Summary:** `SymmetricSparseIterate` is still an `AbstractMatrix`, so generic position loops update each off-diagonal coordinate twice.
- **Affected API:** `X[k]`, `length`, `nnz`, generic loops, `shuffle!`, `rand!` and other Base position-by-position writers on `SymmetricSparseIterate`
- **Conditions:** code that treats the owner as an array of positions
- **Expected:** approved 0.6 owner design (OWN-1, OWN-3, ACT-2; acceptance test T09): the owner is not an `AbstractArray`; such calls are undefined
- **Actual:** `for k in eachindex(x); x[k] *= 2; end` turns an off-diagonal `1.0` into `4.0` (positions 2 and 4 are one variable); `x[2]` is the matrix position `(2, 1)`; `length(x)` is 9; `nnz(x)` is 1; `shuffle!` and `rand!` run silently (all documented as unchecked in iterates.md). A loop over `x[i, j]` that visits both triangles also doubles an off-diagonal variable, but the approved design keeps `x[i, j]` (IDX-2), so that loop is documented behavior, not part of this gap
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `4.0`. At `f9030a64`: 6 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  x = SymmetricSparseIterate(Float64, 3); x[2, 1] = 1.0
  for k in eachindex(x); x[k] *= 2; end; x[2, 1]
  ```
- **Classification:** contract gap (M2, OWN-1, OWN-3, ACT-2)
- **Tracking:** owner [#391](https://github.com/mlakolar/ProximalBase.jl/issues/391); related [#192](https://github.com/mlakolar/ProximalBase.jl/issues/192), [#262](https://github.com/mlakolar/ProximalBase.jl/issues/262), [#368](https://github.com/mlakolar/ProximalBase.jl/issues/368)
- **Priority:** P2 — silent double update for generic code
- **Release impact:** Not yet decided (Q3)
- **Resolution criteria:** T09 of `symmetric-sparse-iterate-0.6-acceptance.md` passes, and the `eachindex`, `shuffle!` and `rand!` rows raise

#### LIM-075

- **Summary:** `copyto!` into views, `vec` and reshapes of a `SymmetricSparseIterate` is unchecked: the upper position wins and pairs collapse.
- **Affected API:** `copyto!(view(X, …), src)`, `copyto!(vec(X), src)`, reshapes and nested wrappers
- **Conditions:** asymmetric source
- **Expected:** the symmetric-matrix rule (M2): checked, or `ArgumentError` before writing
- **Actual:** no error; `copyto!(vec(X), 1:9)` leaves `[1 4 7; 4 5 8; 7 8 9]`, and `reshape(X, 9)` and `view(vec(X), 1:9)` do the same; `copyto!(view(X, 1:2, 1:2), [1 2; 3 4])` stores `[1 2; 2 4]`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[1 4 7; 4 5 8; 7 8 9]`. At `f9030a64`: 4 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  x = SymmetricSparseIterate(Float64, 3); copyto!(vec(x), collect(1.0:9.0)); Matrix(x)
  ```
- **Classification:** contract gap (M2)
- **Tracking:** owner [#368](https://github.com/mlakolar/ProximalBase.jl/issues/368); related [#391](https://github.com/mlakolar/ProximalBase.jl/issues/391)
- **Priority:** P2 — silent data collapse through a route documented as unchecked
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** an asymmetric source through these routes raises before writing

#### LIM-076

- **Summary:** `fill!(x, 0)` and `x .= 0` empty the working set, while `x .= 0 .* x` and `rmul!(x, 0)` keep it.
- **Affected API:** `SparseIterate`, `AtomIterate`
- **Conditions:** an iterate with active coordinates
- **Expected:** FW-2 (approved 0.6 write rule): a value write never removes a coordinate; clearing is `dropzeros!(fill!(x, 0))`
- **Actual:** the working set is emptied
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `Int64[]`. At `f9030a64`: 4 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  x = SparseIterate(4); x[1] = 1.0; x[2] = 2.0; fill!(x, 0.0); activecoordinates(x)
  ```
- **Classification:** contract gap (FW-2, FW-3)
- **Tracking:** owner [#424](https://github.com/mlakolar/ProximalBase.jl/issues/424); related [#456](https://github.com/mlakolar/ProximalBase.jl/issues/456), [#421](https://github.com/mlakolar/ProximalBase.jl/issues/421)
- **Priority:** P2 — the working set depends on spelling
- **Release impact:** Not yet decided (Q3)
- **Resolution criteria:** `fill!(x, 0)` keeps the working set

#### LIM-077

- **Summary:** `coordinates(::SparseIterate)` returns the iterate itself, not a coordinate-vector view.
- **Affected API:** `coordinates(x::SparseIterate)`
- **Conditions:** any `SparseIterate` (a matrix for `N = 2`)
- **Expected:** FC-3 / FD-5 (approved 0.6 target): an `AbstractVector` view of length `ncoordinates(x)`
- **Actual:** `coordinates(M) === M`
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `true`. At `f9030a64`: 2 cases in `reproduce-active.jl`, all `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  M = SparseIterate(Float64, 2, 2); coordinates(M) === M
  ```
- **Classification:** contract gap (FC-3, FD-5)
- **Tracking:** owner [#425](https://github.com/mlakolar/ProximalBase.jl/issues/425); related [#456](https://github.com/mlakolar/ProximalBase.jl/issues/456)
- **Priority:** P3 — API shape of a 0.6 target
- **Release impact:** Not yet decided (Q3)
- **Resolution criteria:** `coordinates(M)` is a length-`ncoordinates` vector view

#### LIM-078

- **Summary:** The vector groups of an `AtomIterate` are shared mutable state: writing through `parentindices(atom(x, k))` desynchronises the layout.
- **Affected API:** `parentindices(atom(x, k))` of `AtomIterate` with vector groups
- **Conditions:** a write into the returned index vector
- **Expected:** FC-6 / FD-6 / FD-11 (approved 0.6 design): the layout is read-only; `atomindices` is the supported read
- **Actual:** `x.groups` changes while `group_of` does not (`atomindices` is not on master; PR #551 is open)
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `[[3, 2], [3, 4]]`. At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  x = AtomIterate(Float64, 4, [[1, 2], [3, 4]]); parentindices(atom(x, 1))[1][1] = 3; x.groups
  ```
- **Classification:** contract gap (FC-6, FD-6, FD-11)
- **Tracking:** owner [#426](https://github.com/mlakolar/ProximalBase.jl/issues/426); related [#422](https://github.com/mlakolar/ProximalBase.jl/issues/422), [#551](https://github.com/mlakolar/ProximalBase.jl/issues/551)
- **Priority:** P3 — internal mutation through an unofficial route
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** the layout cannot be mutated through `parentindices`

#### LIM-080

- **Summary:** `Testing.test_gradient` checks the structure and block sizes of a tuple gradient but not the element type of each leaf.
- **Affected API:** `ProximalBase.Testing.test_gradient` with tuple variables
- **Conditions:** a function whose allocating gradient returns leaves of another element type
- **Expected:** T1: gradient leaves have the element type of the argument; the helper should detect a mismatch
- **Actual:** passes (9 checks, none fail) for `Float64` leaves on a `Float32` tuple
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `test_gradient` passes. At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  # a wrapper whose gradient(f, x::Tuple) returns Float64 leaves; see check LIM-080 in reproduce-active.jl
  ```
- **Classification:** contract gap (T1 (testing helper))
- **Tracking:** owner [#159](https://github.com/mlakolar/ProximalBase.jl/issues/159); related [#546](https://github.com/mlakolar/ProximalBase.jl/issues/546)
- **Priority:** P3 — test helper only
- **Release impact:** Proposed: candidate for deferral
- **Resolution criteria:** `test_gradient` fails on that wrapper

## Approved limitations

Each entry is permitted by the decision it names. They are not defects; a change to any of them needs a new decision.

#### LIM-081

- **Summary:** `IndIntersection` is an approximate projection; `value_and_prox!` returns `0`, which does not certify feasibility.
- **Affected API:** `IndIntersection` (Dykstra's algorithm)
- **Approval:** O3.1, decision 29 (Decided 2026-11-01)
- **Note:** A finite iteration can stop short of the intersection; the returned value is the value at the exact projection. Test the violation separately with the function's tolerance. The fixed default tolerance is a separate gap (LIM-068).
- **Related:** [#138](https://github.com/mlakolar/ProximalBase.jl/issues/138), [#143](https://github.com/mlakolar/ProximalBase.jl/issues/143)

#### LIM-082

- **Summary:** `LogDetLoss`'s fused value uses the mapped eigenvalues, so it can be finite when the stored output rounds to a singular matrix.
- **Affected API:** `value_and_prox!` of `LogDetLoss`
- **Approval:** O3.2, decision 39 (Decided 2026-11-01): "a documented transitional exception until implemented"
- **Note:** Transitional: numerics.md says "until 0.6 is complete"; the target is the value of the stored output (`+Inf32` for the fixture `S = 0`, `X = [0 2^15; 2^15 0]`, `Float32` output). Whether 0.6 ships with it is open (see Q2).
- **Related:** [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489)

#### LIM-083

- **Summary:** Wrappers whose transformed argument overflows are not covered by R1, except norms inside `Translate` and `Precompose`.
- **Affected API:** values of `Translate`, `Precompose`, `LinearImage`, `Tilt`, `Regularize` around non-homogeneous inner functions; user functions inside `Scaled` when their own value overflows
- **Approval:** R1 (wrappers), S4's one stated limitation, decision 45 (Decided 2026-11-01)
- **Note:** The norm case is not covered by this approval: its failure is LIM-023.
- **Related:** [#357](https://github.com/mlakolar/ProximalBase.jl/issues/357), [#490](https://github.com/mlakolar/ProximalBase.jl/issues/490)

#### LIM-084

- **Summary:** In mixed precision, `Translate` and `Tilt` round to the output type between stages, so `prox!` and `prox_entry` can differ.
- **Affected API:** array `prox!` of `Translate`, `Tilt` (nested)
- **Approval:** O4, decision 30 (Decided 2026-11-01)
- **Note:** Lipschitz inner maps carry a per-stage bound; discontinuous inner maps promise the staged model and its tie rule.
- **Related:** [#490](https://github.com/mlakolar/ProximalBase.jl/issues/490)

#### LIM-085

- **Summary:** Matrix reconstructions may saturate entries at `floatmax` within their normwise error bound.
- **Affected API:** spectral reconstructions on the scaled path
- **Approval:** R4 (last sentence), decision 39 (Decided 2026-11-01)
- **Note:** R4's no-saturation rule applies to entrywise results only.
- **Related:** [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489)

#### LIM-086

- **Summary:** The summation order of iterate products is unspecified, so a finite overflow can depend on the order (`dot(x, y) == Inf`, `dot(y, x) == M`).
- **Affected API:** `dot`, `mul!` over iterates
- **Approval:** decision 15 (Decided 2026-11-01: "Sums may be reordered."), stated in numerics A6; numerics R2 sends the outputs of matrix products to A1, and the iterate design FN-4 (signed off with FD-9, #400/#402) states the consequence for iterate products
- **Note:** Only the class of zeros and non-finite entries is promised (FT12); strong zeros themselves are LIM-071.
- **Related:** [#428](https://github.com/mlakolar/ProximalBase.jl/issues/428)

#### LIM-087

- **Summary:** Zero scaling is rejected where no domain-preserving proximal representation exists.
- **Affected API:** `Scaled(NegLog(), 0)`, infinite-level functions, `LogDetLoss` (finite positive weight)
- **Approval:** E4, decision 40 (Decided 2026-11-01)
- **Note:** Total-domain functions that are rejected today are a gap (LIM-058).
- **Related:** [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485)

## Housekeeping

Defects that no rule of the contract covers. They are listed so that they are not lost, and checked by `reproduce-active.jl`, but they are not contract gaps.

#### LIM-079

- **Summary:** The asymmetry message prints a NaN relative asymmetry for `BigFloat` entries beyond the `Float64` range.
- **Affected API:** `copyto!` and other symmetric-iterate writes with an asymmetric `BigFloat` source
- **Conditions:** entries above `floatmax(Float64)`
- **Expected:** a finite relative asymmetry in the message (the exception and its class are right)
- **Actual:** "(relative asymmetry NaN)"
- **Evidence:** at `920ebece`, same on Julia 1.13.3 and 1.10.12: `ArgumentError … (relative asymmetry NaN)`. At `f9030a64`: 1 case in `reproduce-active.jl`, `PRESENT` on Julia 1.13.3 and 1.10.12.
  ```julia
  copyto!(SymmetricSparseIterate(BigFloat, 2), [big"1e400" big"2e400"; big"3e400" big"1e400"])
  ```
- **Classification:** housekeeping: no contract rule applies to the text of a diagnostic (moved out of the contract gaps after the review of #558)
- **Tracking:** owner [#391](https://github.com/mlakolar/ProximalBase.jl/issues/391); related [#415](https://github.com/mlakolar/ProximalBase.jl/issues/415)
- **Priority:** P3 — cosmetic
- **Release impact:** none (not a contract gap)
- **Resolution criteria:** the message reports a finite relative asymmetry

## Intentional exclusions

The contract or the user documentation excludes these on purpose, and the
calls raise before writing (or are documented as unchecked). They are listed
so that they are not reported again as defects. Where an exclusion is
enforced with the wrong exception class, the class is a gap above.

| excluded | where it is stated | related gap |
|:---|:---|:---|
| complex arguments, outputs, steps and parameters | numerics S1, interface.md (Precision) | LIM-010, LIM-011, LIM-012 |
| automatic-differentiation numbers | numerics S1 | — |
| `BigFloat` and `Float16` in the LAPACK families (`IndPSD`, `NuclearNorm`, `LogDetLoss`, `IndStiefel`, spectral wrappers) | numerics T6, T7, decision 8; spectral docstrings | LIM-046 (argument type unchecked) |
| output–input sharing other than `y === x` (shifted or reversed views, distinct views of the same entries, overlapping gradient or Hessian outputs, `gradient_mapping!`, `lmo!`) | numerics O7, decision 32 (clarified 2026-11-03) | LIM-091 (parameters, undecided) |
| array and tuple steps in `SeparableSum`, `SliceSum`, `FiberSum` | block-sum docstrings | — |
| array steps for cone indicators, `SqrDistL2`, `DistL2`, `PointwiseMinimum`; within-group metrics for group norms; non-positive or non-finite array steps on coupling sets | sets.md, calculus.md, interface.md, numerics P1 | — |
| `Constrained` pairs outside the documented list; a box that misses `dom f` | `Constrained` docstring, calculus.md | — |
| `LinearImage` beyond its two cases; `Spectral` with entry-dependent parameters; `SingularValue` of a function that is not absolutely symmetric | docstrings | — |
| `restrict` of `NegLog`, of boxes that exclude zero, of nonseparable `Translate` with a translation outside `S`, of `MoreauEnvelope`/`SqrDistL2` of nonseparable functions, of array-scaled group penalties; `BernoulliLoss` | `restrict` and `Precompose` docstrings | LIM-066 |
| on device arrays: sparse iterates, GLM second-order oracles, `LeastSquaresLoss` prox, `IndBallGroupL0` with `k`, wrapped device storage for spectral functions; decompositions run on the host | interface.md (GPU arrays) | LIM-060 to LIM-063 |
| one function object per thread; group arrays and their indices must not be mutated after construction | interface.md, iterates.md | LIM-106 (concurrent reads, unverified) |
| values that are symmetric only up to rounding are rejected; raw entry oracles do no symmetry check; group penalties, cones and fiber sums on a `SymmetricSparseIterate` | iterates.md (symmetric-matrix rule), decision 33; #342 | LIM-105 |
| the standalone value of most conjugates; `PointwiseMinimum`'s value without component values | calculus.md, `PointwiseMinimum` docstring | LIM-064 |
| `lmo!` outputs other than dense arrays; sets beyond the documented list | `lmo!` docstring | — |
| `ConvolutionCheck` kernels other than `:uniform` and `:epanechnikov`; LLA weights of `NormLq` | docstrings, penalties.md | — |
| `Symmetric`/`Hermitian` outputs of `Translate`, `Tilt`, `Regularize` | composition docstrings | LIM-008 (other functions) |
| other `BallConstrained` inner functions in E3's determined-limit formulas | numerics E3 (closed list), decision 48 | LIM-055 |

## Coverage unverified

Status not established. These are not confirmed defects; each says why, and what would settle it.

| ID | behavior | affected API | why unverified | what would settle it | owner |
|:---|:---|:---|:---|:---|:---|
| <a id="lim-088"></a>LIM-088 | Real GPU hardware is not tested: device kernels, compensated product paths, cold ball/sphere paths and user functions with array fields run only on JLArrays (CPU-executed) in CI. | GPUArrays extension (all device kernels) | no CUDA/ROCm/Metal hardware; JLArrays checks dispatch and kernel logic, not flush-to-zero, fast-math, compilation or synchronisation (G3, G6) | an opt-in discovery run (`test/cuda`) on real hardware with recorded results | [#494](https://github.com/mlakolar/ProximalBase.jl/issues/494) |
| <a id="lim-089"></a>LIM-089 | Several functions with array parameters may lack Adapt rules (`SqrHingeLoss`, `IndBinary`, `BernoulliLoss`, weighted `CappedL1`/`LogPenalty`/`NormLq`/`Berhu`, `Tilt`); JLArrays hides the gap because its kernels run on the CPU. | `adapt(CuArray, f)` for these functions | found by code reading; JLArrays cannot show a missing device copy | a real-device run, or a test that checks the adapted fields are device arrays | [#494](https://github.com/mlakolar/ProximalBase.jl/issues/494) |
| <a id="lim-090"></a>LIM-090 | `prox_coordinate!` of regularised group penalties and `gradient_mapping!` leave a partial but consistent state when a user-supplied array throws. | `prox_coordinate!`, `gradient_mapping!` | O6 has no category for failures inside user code during a coordinate update | a maintainer decision on the O6 category, then a test of the decided state | [#495](https://github.com/mlakolar/ProximalBase.jl/issues/495) |
| <a id="lim-091"></a>LIM-091 | A parameter or step array that is also the output (`y === b` for `Translate`, `y === c` for `Tilt`, `y === γ`) is not covered by O7; `Translate` with `y === b` returns wrong values. | `Translate`, `Tilt`, array steps, weights, centres | O7 speaks only of output–argument sharing; the rule for parameters is unstated | a decision in #367 and a test of the decided behavior | [#367](https://github.com/mlakolar/ProximalBase.jl/issues/367) |
| <a id="lim-092"></a>LIM-092 | Writes that do not convert exactly into an integer iterate from dense sources (`copyto!`, broadcast, `map!`, `rmul!`) raise `InexactError` after earlier entries were written; the state stays consistent. | `copyto!`, broadcast, `map!`, `rmul!` into `SparseIterate{Int}`, `SymmetricSparseIterate{Int}` | decision 8 rejects integer destinations for fractional operations, but O6 has no category for value-dependent conversion failures at write time | a decision (check every conversion first, or document partial-but-consistent), then tests | [#495](https://github.com/mlakolar/ProximalBase.jl/issues/495) |
| <a id="lim-093"></a>LIM-093 | Matrix-product kernels form the unscaled product before `α < 1` rescales it (`mul!` into a `SymmetricSparseIterate`, `_SymmetricPart`). | `mul!(C, S, B, α, β)` paths | contract ambiguity: A1 (matrix products) versus R2 for an overflowing intermediate with a representable result; behavior also depends on the BLAS | an answer to the two questions in the #366 comment of 2026-11-01 (#404 note) | [#366](https://github.com/mlakolar/ProximalBase.jl/issues/366) |
| <a id="lim-094"></a>LIM-094 | Gradients of `LeastSquaresLoss`, `QuadraticFunction` and `KronQuadratic` can overflow in matrix products at representable results. | `gradient!` of data losses | contract ambiguity: matrix products follow A1, not R2; #542 collects the cases | a stated A1 bound and tests, or a decision that R2 applies | [#542](https://github.com/mlakolar/ProximalBase.jl/issues/542) |
| <a id="lim-095"></a>LIM-095 | The `Conjugate` support-function fallback evaluates a user's oracle in `eltype(y)`, so a set bound outside `Float32` range is rounded to `Inf`. | `Conjugate` of user-defined sets with an `lmo!` | depends on user oracles; T3 for user code is not settled | a decision on the precision of user oracles (#187) | [#187](https://github.com/mlakolar/ProximalBase.jl/issues/187) |
| <a id="lim-096"></a>LIM-096 | Proximal outputs of `Spectral(f)` and `SingularValue(f)` at an overflowing spectrum, for `f` without a scaling rule (`SqrNormL2`, `CubeNormL2`, `NormLq`, `NormL0`, bounded balls). | `prox!` of spectral wrappers | contract ambiguity: decision 45's wrapper limitation versus R2 for built-in inner functions; PR #384 holds the open questions | the #384 decisions A/B and tests | [#283](https://github.com/mlakolar/ProximalBase.jl/issues/283) |
| <a id="lim-097"></a>LIM-097 | `IndGraph(A)` construction raises for entries near `2^27` or an overflowing Gram matrix (no scaling). | `IndGraph` constructor | contract ambiguity: A1 for factorizations versus S4 range expectations | a stated range for data functions (G4) | [#487](https://github.com/mlakolar/ProximalBase.jl/issues/487) |
| <a id="lim-098"></a>LIM-098 | `IndAffine` truncates the rank at `max(m, n)·eps·σ₁` and uses an `eps^(1/4)` pivot test, so a nearly rank-deficient `A` is replaced by its truncation. | `IndAffine` | A4's rank convention (`√eps·σ₁`) and a solver-style rank cutoff disagree; which applies to a data function is not decided | a G10 decision on rank tolerances for data | [#493](https://github.com/mlakolar/ProximalBase.jl/issues/493) |
| <a id="lim-099"></a>LIM-099 | Off the positive-definite cone, `LogDetLoss`'s `hessian_vec!` returns NaN while `quadratic_model` raises. | `hessian_vec!`, `quadratic_model` of `LogDetLoss` | E6 / decision 44 cover gradients (NaN); second-order oracles outside the domain are not decided | a decision extending E6 to second-order oracles | [#491](https://github.com/mlakolar/ProximalBase.jl/issues/491) |
| <a id="lim-100"></a>LIM-100 | An infinite level with a zero step: `NormL1(Inf)` gives the identity, `ElasticNet(Inf, 1)` gives NaN. | `prox!` at `γ = 0` with an infinite level | E4 says an infinite level is not an infinite weight unless the function says so; neither function says | a per-family statement and a test | [#486](https://github.com/mlakolar/ProximalBase.jl/issues/486) |
| <a id="lim-101"></a>LIM-101 | An infinite entry is a member of an unbounded box: `IndNonnegative()([Inf]) == 0`, `IndBox(-Inf, Inf)([Inf]) == 0`. | values of `IndNonnegative`, `IndBox` with infinite bounds | contract ambiguity: E3's sentence gives `+Inf` for an indicator of real vectors, but an extended-real box contains `Inf`; `test/separable.jl:413` pins `0` | a decision on which reading applies | [#485](https://github.com/mlakolar/ProximalBase.jl/issues/485) |
| <a id="lim-102"></a>LIM-102 | Five-argument `mul!` into a `SymmetricSparseIterate` with `α = Inf` and an empty inner dimension gives NaN where the dense `mul!` gives zeros. | `mul!(X::SymmetricSparseIterate, A, B, Inf, β)` | E4/E2 for scalar coefficients of products is not stated | a decision (with the #391 products) and a test | [#391](https://github.com/mlakolar/ProximalBase.jl/issues/391) |
| <a id="lim-103"></a>LIM-103 | The class (finite, `Inf`, NaN) of `Symmetric * X * Symmetric` for a sparse iterate `X` depends on density and association. | three-factor products with iterates | open decision FD-16 (#419) | the #419 decision and its FT11 rows | [#419](https://github.com/mlakolar/ProximalBase.jl/issues/419) |
| <a id="lim-104"></a>LIM-104 | `NormL0` compares the rounded square and threshold (`x² > 2γλ`); 165 of 3,000 `Float64` near-tie samples differ from the exact decision. | `prox!` of `NormL0` (and `BigFloat` calls) | contract ambiguity: whether A2's "model's operands" means the exact or the rounded expression | a decision on the comparison A2 requires for squared thresholds | [#519](https://github.com/mlakolar/ProximalBase.jl/issues/519) |
| <a id="lim-105"></a>LIM-105 | `FiberSum(f; symmetric = true)` reads only the fibers with `i ≥ j` of an asymmetric input and copies them to `(j, i)`. | `FiberSum` with `symmetric = true` | the symmetry policy for block sums is undecided (#342) | the #342 policy and a test | [#342](https://github.com/mlakolar/ProximalBase.jl/issues/342) |
| <a id="lim-106"></a>LIM-106 | Value calls of group penalties write the layout cache into the `AtomIterate`; concurrent value calls on one iterate are not specified. | values of group penalties on `AtomIterate` | M3 and the one-object-per-thread rule cover function objects, not concurrent reads of one iterate | a stated rule for concurrent reads, or a cache that is not written on read | [#456](https://github.com/mlakolar/ProximalBase.jl/issues/456) |
| <a id="lim-107"></a>LIM-107 | `Spectral(ElasticNet)` and `SingularValue` were not audited for the T3/R2 threshold-product changes of #503 and #519. | spectral wrappers of weighted penalties | not covered by the audits of #503/#519 | a T3/R2 sweep of the spectral wrappers with BigFloat references | [#489](https://github.com/mlakolar/ProximalBase.jl/issues/489) |
| <a id="lim-108"></a>LIM-108 | `Testing.test_separable` skips the zero-value and sparse value checks silently when the value method does not accept the argument. | `ProximalBase.Testing.test_separable` | established by code reading and #524's description; not executed in this audit | a test of the helper with a vector-only value method and a matrix argument | [#524](https://github.com/mlakolar/ProximalBase.jl/issues/524) |

## Resolved since the audit

When an entry is fixed, it is listed here as `ID — fixed by #PR (commit), evidence`.

- <a id="lim-034"></a>**LIM-034** — nested `Scaled` turned an underflowing product of finite scales into `ZeroFunction` and an overflowing one into `Inf` (T5, R3). Fixed by [#549](https://github.com/mlakolar/ProximalBase.jl/issues/549) (`f9030a64`, 2026-11-04), after the audit. At `f9030a64` all 7 cases of the entry printed `FIXED` on Julia 1.13.3 and 1.10.12: `Scaled(Scaled(NormL1(), 1e-200), 1e-200)` stays a nested `Scaled` with value `1e-100` at `[1e300]`, the overflowing pair keeps its finite factors (value `1e100` at `[1e-300]`), and its prox at `[3e-100]`, `γ = 1e300` is `[2e-100]`. Regression tests: `test/nested_scaled.jl` lines 47, 53, 68, 89–90 and 212–215. The cases were then removed from `reproduce-active.jl`. [#333](https://github.com/mlakolar/ProximalBase.jl/issues/333) stays open with the follow-ups [#566](https://github.com/mlakolar/ProximalBase.jl/issues/566), [#567](https://github.com/mlakolar/ProximalBase.jl/issues/567) (LIM-024) and [#568](https://github.com/mlakolar/ProximalBase.jl/issues/568).
