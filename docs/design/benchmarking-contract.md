# Cross-package benchmarking contract (proposed)

> **Status: draft for maintainer review (2026-11-03). Not signed off, not a claim of existing implementation, and not a release requirement by itself.**
>
> **Canonical design location:** ProximalBase.jl, `docs/design/benchmarking-contract.md`.
> **Intended adopters:** ProximalBase.jl (kernel primitives), ProximalSolvers.jl (optimization algorithms), and HighDimensionalModels.jl (statistical estimators). Other packages may adopt an appropriate subset later.
>
> **Coordination:** [ProximalBase #514](https://github.com/mlakolar/ProximalBase.jl/issues/514),
> [ProximalSolvers #110](https://github.com/mlakolar/ProximalSolvers.jl/issues/110),
> [HighDimensionalModels #216](https://github.com/mlakolar/HighDimensionalModels.jl/issues/216).
> Correctness CI and the 0.6 compatibility migration are tracked **separately**:
> [ProximalSolvers #121](https://github.com/mlakolar/ProximalSolvers.jl/issues/121),
> [HighDimensionalModels #195](https://github.com/mlakolar/HighDimensionalModels.jl/issues/195),
> [ProximalBase #527](https://github.com/mlakolar/ProximalBase.jl/issues/527).

This document specifies the **desired eventual behavior** of the package-family benchmarking workflow. It defines common terms and evidence standards, while leaving workloads, numerical certificates, measurement engines, and justified tolerances to each package. It deliberately does **not** mandate a new shared library, a full performance test on every pull request, or a common timing engine.

## 1. Design goals, boundaries, and interpretation

The leading metric is **time to a trustworthy development decision**, not how many benchmark cases can be run or how impressive a dashboard looks. Routine edits need quick feedback; performance claims need calibrated evidence.

Three distinct activities share a catalog and evidence conventions:

| Activity | Decision it supports | Evidence level | Time budget |
|:--|:--|:--|:--|
| **Explore** | Is an optimization worth investigating or implementing? | Fast, preliminary correctness/accuracy and timing | As short as practical; warm-loop seconds or tens of seconds are *targets*, not guaranteed performance |
| **Confirm** | Is this selected candidate faster, or demonstrably noninferior within a predeclared tolerance? | Fixed-workload, pinned, independently repeated A/B with uncertainty | Higher, explicitly budgeted cost on appropriate hardware |
| **PR / release review** | Is a proposed change or dependency migration safe to merge/release? | Correctness CI by default; targeted **Confirm** evidence when warranted; broader campaigns periodically | Risk-dependent; never require full extended performance benchmarking on ordinary PRs |

Statistical evidence is a **method**, not a CI tier: a local investigation may invoke Confirm, and a performance-sensitive PR may reuse a valid confirmed comparison. An ordinary green test job is neither evidence of speedup nor proof of noninferiority.

**Normative language.** Rules below are *proposed contract requirements*: **MUST** is a condition for a package/workflow to claim conformance **after it explicitly adopts this contract**; **SHOULD** is a default that may be replaced with a documented, evidence-backed alternative; **MAY** is optional. Existing package behavior is not retroactively nonconforming. Rule IDs (`BC-01`, etc.) remain stable when implementation guidance changes.

**Non-goals:** optimizing individual algorithms; modifying mathematical/numerical contracts; changing stopping criteria to make timings look faster; comparing arbitrary unpinned study environments; guaranteed detection of a true 5% slowdown; building a separate benchmark daemon; a central service or new Julia package before proven reuse; replacing existing benchmark records with an untested schema; turning statistical estimation-quality studies (coverage, FDR, power, prediction error) into timing gates.

## 2. Package boundary: share policy, keep domain-specific measurements

| Layer | Primary quantity | Suitable measurement | Local correctness gate |
|:--|:--|:--|:--|
| **ProximalBase** | Cost and allocation of a mathematical kernel/operator | `BenchmarkTools.jl`, including state preparation and explicit evaluation policy | Independently checked oracle/output and structural/zero-allocation contracts |
| **ProximalSolvers** | Time to a specified optimization accuracy; separately, setup and reused workspace cost | Existing `@timed` whole-call/phase runner; diagnostic iteration and prox/gradient counts | Independent numerical certificate of returned solution; successful stopping flags alone are insufficient |
| **HighDimensionalModels** | Time of the public estimator or path, with relevant estimator setup and post-processing | Existing `@timed` public API runner; explicit workload families | Independent estimator/optimization certificate or a clearly identified diagnostic-only case |

**BC-01 — Local ownership.** Each package MUST own its case definitions, accurate fixture construction, domain-specific numerical references/certificates, and what exactly is timed. An inference report MUST NOT silently pass a solver/estimator that failed its accuracy target.

**BC-02 — Shared conventions.** Packages adopting this contract MUST use consistent meanings for *exploratory*, *confirmed*, *primary*, *diagnostic*, *A/A*, *baseline/candidate*, *improvement*, *noninferiority*, *investigate*, *inconclusive*, and the after/before timing ratio. A shared CLI/reporting adapter MAY be developed; a shared implementation dependency is **not** required.

**BC-03 — No forced measurement-engine migration.** ProximalBase MAY continue using `BenchmarkTools` for tiny kernels; ProximalSolvers/HDM MAY continue using `@timed` for full solves. The shared layer MUST NOT interpret samples from materially different measurement methods as directly interchangeable, or mistake fast kernel calls for fast time-to-solution.

## 3. Benchmark case and validity contract

A *case* is a stable name plus a specific mathematical workload, data regime, operation/phase, precision, state/reset policy, metric, independent validation, and declared budget. Its semantic meaning must remain explicit as the source implementation evolves.

**BC-04 — Case identity.** Each benchmark case MUST have a stable package-scoped ID and metadata describing its group/family, operation, phase, input type/shape or workload regime, and any relevant numerical target. Keep case IDs stable across revisions when semantics are stable; give a changed meaning a versioned/new identity, or document it and invalidate prior baselines. IDs need not share a forced grammar across packages.

**BC-05 — Deterministic scenarios.** Source revisions compared formally MUST use identical meaningfully keyed scenarios/seed schemes and comparable problem data; data hashes and work units MUST be verified when available. Selecting one case instead of a large group MUST NOT change that case's numerical input, intended initial state, or certificate. Cases that intentionally vary inputs across fresh-process repetitions MUST pair matching scenario keys on both sides.

**BC-06 — Independent validation.** Before timing can be treated as successful, each case MUST independently establish that the measured operation did the intended work. References MUST not be derived from the potentially corrupted output or from the method under test. For solver/estimator cases, independently certify the achieved accuracy using declared mathematical criteria; a returned `converged=true` alone is insufficient. For nonconvex cases, distinguish stationarity certificates from global-optimality guarantees; for estimator selection/post-processing, state exactly which properties are checked.

**BC-07 — Valid warm and reset states.** Cases MUST declare whether they measure fresh construction, first activation, repeated warm calls, or a reusable workspace. Mutating operations MUST be reset to the same intended state before each valid observation. Compilation, warm-up, fixture generation, and independent certification MUST be outside the timed region **unless that work is explicitly part of the named public-operation estimand**. Reset and setup costs MUST NOT be silently subtracted from a timed figure when that subtraction changes the estimand.

**BC-08 — Failure preservation.** Every attempted observation MUST retain its outcome (success, numerical target miss, uncertifiable, incomplete, timeout, exception, and any package-specific status). Failed or unverified results MUST NOT be silently removed to calculate a faster mean, and an all-failed case MUST NOT be represented as a successful timing. Record enough diagnostic information to understand missing and invalid comparisons.

**BC-09 — Workload relevance.** Representative cases SHOULD cover the major dimensions that affect performance: size and scaling; Float32/Float64; dense/sparse structures; first use versus warm calls; allocations and workspace reuse; important degenerate/boundary workloads. Not every cross-product belongs in routine PR checks. Select **primary** cases based on affected code and plausible regressions, not on cases that happened to show a favorable ratio or low observed variance.

**BC-10 — Distinct non-timing contracts.** Correctness, numerical behavior, allocation ceilings, complexity obligations, and (for GPU) device synchronization/transfer semantics are independent of statistical timing verdicts. A timing pass MUST NOT override a correctness, certificate, or agreed allocation failure. Julia heap bytes are not native-library, accelerator, or peak-memory bytes; reports MUST label these distinctions.

## 4. Explore: fast local optimization workflow

**BC-11 — Cheap first signal.** The default local workflow MUST permit selection of a small number of relevant cases, fast integrity checks, and short preliminary timings *without running a full test suite or extended performance campaign*. Its usefulness MUST be assessed by **end-to-end latency from edit/invocation to first actionable result**, not timed-kernel duration alone.

**BC-12 — Honest preliminary claims.** Exploratory output MUST be labeled **exploratory**, including when it comes from a warm Julia session or only one source execution. It MAY show absolute times, allocation observations, estimated ratios, accuracy results, and iteration counts. It MUST NOT claim a statistically established improvement, a proven absence of regression, or noninferiority based solely on that output.

**BC-13 — Session integrity.** Persistent Julia sessions, `Revise`, `TestEnv`, or cached fixtures MAY be used if they measurably help. They MUST detect/handle stale source and reset shared state correctly; a changed source or mutable fixture that is not actually executed is not a valid benchmark. Fresh-process runs remain the authority for formal revision comparisons. A warm-session optimization MUST be explicitly tested with a deliberate changed method and a deliberately broken numeric outcome.

**BC-14 — Measured iteration gains.** Every local-workflow infrastructure change SHOULD report cold and warm startup, package loading, compilation, fixture construction, validation, execution, and reporting overhead, plus full time to first useful feedback. The objective is less work and faster decisions, not more flags or a new daemon. Prefer a simple script and existing case catalog to a competing framework.

Recommended local output:
- selected case ID and workload/phase;
- numerical/reference-check status, with target and achieved value when applicable;
- absolute time, Julia heap bytes, and optionally operation/iteration counts;
- when A/B was attempted, observed candidate/baseline ratio and source identities;
- a prominent warning that ordinary local ratios are **not formal statistical evidence**.

## 5. Confirm: paired statistical comparison

**BC-15 — Frozen hypothesis and primary set.** Before a confirmatory campaign, record the candidate/base revisions, specific *primary* cases, independent accuracy/allocations criteria, expected performance benefit or noninferiority margin, statistic per fresh process, permitted repetitions, and comparison protocol. Additional cases may be diagnostic; do not retroactively select primary cases by their observed result.

**BC-16 — Pinned comparable environments.** Ordinary within-package confirmation MUST use the same fixed benchmark harness and case meaning, comparable package dependencies, Julia/BLAS versions, CPU/OS/architecture, bounds-check/instrumentation settings, thread counts, seed scheme, and measurement budgets. Pin both source identities and a manifest/content hash. A clean/dirty tree and the exact executed package root MUST be recorded. **Unknown or different settings are not silently “matched.”** Reject an invalid comparison or explicitly classify it under the separate migration protocol in `8.

**BC-17 — Fresh-process paired design.** Formal estimates MUST be based on independent *process-level paired repetitions* (or an explicitly validated alternative) with corresponding input scenarios. Alternate execution order (AB, BA, ...) on the same reasonably isolated machine, and retain the order log. Do not run competing timing jobs on the same timing cores. Calibration, compilation, warm-up, setup, and certification requirements remain per-case. Do not use a shared GitHub Actions host's ordinary wall-clock ratios as a mandatory performance gate without calibration of that environment.

**BC-18 — Correct inference unit and estimand.** Let the declared per-process case statistic be `T_baseline,r` and `T_candidate,r` (for example a BenchmarkTools run-level mean in ProximalBase, a run-level median of certified whole-solve times in ProximalSolvers/HDM). Define

$$
d_r = \log\left(\frac{T_{\mathrm{candidate},r}}{T_{\mathrm{baseline},r}}\right),
\qquad \widehat{\rho}=\exp\left(\operatorname{mean}_r d_r\right).
$$

The timing ratio `rho` is **candidate/baseline**; smaller is faster. The reported statistic and any transformation MUST be declared before comparison. Individual within-process BenchmarkTools samples or repeated `@timed` observations are **not** independent process-level inferential units. Do not automatically pool observations from different Julia/compiler/architecture strata.

**BC-19 — Margins and evidence classes.** Every formal comparison MUST separate these distinct claims:

| Claim | Proposed criterion using one-sided `1-alpha` confidence bounds for `rho` | Interpretation |
|:--|:--|:--|
| **Improvement established** | Upper bound `U < 1` (or `U < 1 - epsilon` for a prespecified practically important improvement) | Evidence that the selected candidate is genuinely faster on this case |
| **Noninferiority established** | `U < 1 + delta`, with an approved, prespecified `delta >= 0` | Evidence the slowdown is smaller than the allowed margin |
| **Material regression established** | Lower bound `L > 1 + delta` | Evidence that the slowdown exceeds the approved margin |
| **Investigate / warning** | A predeclared rule flags plausible slowdown but a material-regression claim is not established | Diagnose independently, do not label as proof of a >margin slowdown |
| **Inconclusive** | No applicable bound establishes a claimed outcome, or validity/calibration is insufficient | Neither an automatic pass nor a demonstrated regression |

Improvement implies noninferiority when `delta >= 0`; output SHOULD report both the **claim being evaluated** and all relevant confidence bounds, rather than hiding this overlap. An interval crossing 1 does not by itself imply inferiority, nor does failure to reject slowdown prove noninferiority. A missing/invalid interval is inconclusive, regardless of the point estimate.

For ProximalBase 0.6 hot paths, **retain the existing policy in #462 / PR #499**, unless separately amended by the maintainer: with `delta=0.05`, `pass` when the *one-sided 95% upper bound* is below 1.05; `investigate` when the point estimate is at least 1.05 **and** the one-sided lower bound exceeds 1; otherwise `inconclusive`. **“Investigate” is not proof that the true slowdown exceeds 5%.** Correctness, allocation, and complexity remain separate. Do not assume this 5% margin is calibrated or appropriate for whole solvers or estimators; those owners must justify their own margins.

**BC-20 — Calibration before automatic gating.** A mandatory timing gate MUST first demonstrate independent A/A (unchanged-vs-unchanged) stability under its stated environment and per-case sampling budget. Report false investigate events, interval behavior, distribution of paired ratios, missing/uncertifiable cases, process-order effects, and how frequently unchanged code is found noninferior at the selected margin. A planning goal such as ≥80% probability of noninferiority pass for an unchanged primary case may be used, but it is **not** a guarantee of 80% power to detect a slowdown exactly on the allowed boundary. Where the noise floor exceeds the effect of interest, label the case diagnostic, improve measurement, or use an explicit human decision; do not relax margins to produce green dashboards.

**BC-21 — Selection, repeated looks, and multiplicity.** An agent MAY explore many candidates cheaply, but confirmation of a selected winner MUST use newly collected evidence with fixed primary cases and margins. If repetitions are added while inspecting results, use a prespecified fixed-sample plan or a valid sequential/alpha-spending procedure. Repeated ordinary one-sided 95% looks do not preserve nominal coverage. If a *family-wide* claim is made, address the family-level error claim/multiple comparisons explicitly; do not silently turn many independent per-case tests or a family geometric mean into a single hard merge gate. Do not selectively drop “bad” samples, executions, or measurements after observing an unfavorable result.

**BC-22 — Validity overrides timing.** All primary cases MUST first satisfy their accuracy and integrity checks. If a candidate reaches a worse stopping criterion, has an incomplete path, or produces uncertifiable outputs, this is **not** an equivalent workload/time-to-accuracy improvement. Formal reports MUST preserve these failures and may classify the timing comparison as unavailable. Diagnostics such as solver iteration count can help attribute differences, but must not replace independent certification.

## 6. Ordinary PRs, performance-sensitive PRs, and periodic campaigns

**BC-23 — Cheap PR integrity, not default wall-clock gating.** On routine PRs, use the relevant repository's **correctness CI owner** and inexpensive benchmark-analysis/ref-check controls. Select affected cases and execute no more than the needed integrity/certificate checks. Routine PR checks SHOULD be relatively quick and run in parallel where correctness testing permits; timing campaigns themselves SHOULD use isolated resources. Do not make full extended benchmark suites or uncalibrated shared-runner ratios mandatory on every PR.

**BC-24 — Performance-sensitive PR receipt.** A hot-path change, an optimization PR claiming a speedup, or an important suspicious performance change SHOULD have a short evidence receipt containing: changed code path; preregistered primary and diagnostic cases; baseline/candidate commits; fixed harness and dependencies; hardware/settings; independent accuracy targets/outcomes; timing estimand and run-level statistic; process-paired observations and order; interval and margin; allocations; investigated and inconclusive cases; explicit maintainer approval for intentional costs; and a raw-evidence link. The verdict on a correctness PR need not demonstrate improvement; an optimization claim should substantiate a meaningful primary improvement and protect relevant secondary workloads.

**BC-25 — Escalation, not blanket reruns.** If important primary cases are inconclusive, choose an independent follow-up or a documented maintainer review. More repetitions are justified only when the measured noise suggests they could clarify the decision. A long periodic or release campaign SHOULD monitor representative algorithms, sizes, precisions, and architectures and identify regressions for follow-up. It MUST NOT silently overrule the case-specific evidence/meaning.

**BC-26 — CI ownership separation.** ProximalBase #514 tracks performance correctness/calibration; ProximalSolvers #121 and HDM #195 track test-selection/merge-gate policy. A shared benchmark orchestration layer MUST NOT create a competing required status check or change branch protection without its package CI owner's approval.

## 7. Evidence, provenance, artifacts, and portability

**BC-27 — Minimal transferable evidence.** A confirmed comparison MUST preserve enough raw observations to recreate the report and independently verify ratios, intervals, failures, and the stated source/configuration. Common report fields SHOULD be normalized through adapters as necessary; **this contract does not require rewriting existing TSV/TOML/CSV formats or changing their hashing semantics**.

Suggested common concepts (field names and serialization are **proposed**, not an implemented schema):

| Concept | Minimum information |
|:--|:--|
| Case | package, case ID, group, operation/phase, quick/extended, primary/diagnostic, numerical target |
| Experiment | `explore` / `confirm` / `migration`; source and baseline labels, timestamp/campaign ID, prespecified comparison rules |
| Source/environment | exact commit/source hash/dirty flag, harness hash, Julia and package versions, manifest/dependency identities, CPU/OS/architecture, BLAS/Julia threads, bounds/coverage settings, optional CPU-frequency/power policy |
| Replication | side, pair ID, seed/data hash, process identity, run order, sample budget, per-process declared timing summary |
| Observation | time unit (ns preferred), samples, Julia heap bytes, GC when available, accuracy measure/threshold/status, failure diagnostics, optional RSS/device timing |
| Assessment | `rho`, confidence bound type/level, `delta` and/or `epsilon`, A/A calibration reference, improvement/noninferiority/investigate/inconclusive, human review |
| Retention | full raw records, analysis/code revision, reproducible invocation, artifact checksum, report link |

**BC-28 — Honest environment comparisons.** Ordinary performance regressions MUST compare semantically compatible versions of the same workload under pinned environments. For cross-Julia/compiler, cross-architecture, external-competitor, or dependency-migration comparisons, declare the intended **stack-level estimand**. Never weaken strict ordinary-comparison identity checks silently just to produce a ratio.

**BC-29 — Archive hygiene.** Evidence SHOULD be kept as durable artifacts with checksums and documentation links, without committing large generated records. Where release/issue assets are public, scrub hostnames, usernames, sensitive filesystem paths, and other infrastructure details. HDM's established #139 archive rules remain authoritative for its campaign files.

## 8. The ProximalBase 0.6 upgrade is a special experiment

The compatibility obligations in [ProximalBase #527](https://github.com/mlakolar/ProximalBase.jl/issues/527) remain hard release requirements: migrate downstream packages, update compat, and pass agreed cross-package integration tests.

Separately, before migrating, capture a **small, pinned representative pre-0.6 performance baseline** for key solver and estimator workloads. After migration, measure the same meaningful workloads at matched certified accuracy and inspect major unexplained time/allocation changes.

**BC-30 — Migration comparisons are labeled.** The output MUST distinguish the *old entire stack* from the *new entire stack*. If ProximalBase and downstream code both change, a new/old ratio is **not** clean causal attribution to ProximalBase itself. Ordinary comparator matching rules SHOULD remain strict; a dedicated migration comparison may explicitly allow declared dependency differences but MUST record and display them and the attribution limit. Do not demand the completed automation platform to release 0.6. An unresolvable few-percent timing difference on noisy hardware is not automatically a release blocker.

## 9. Tooling architecture: one contract, not necessarily one codebase

Use existing, credible Julia tooling first:

- **Measurement:** retain `BenchmarkTools.jl` for kernel microbenchmarks; retain the existing `@timed` approach for certified solver/estimator whole-call timings. `Chairmarks.jl` is a candidate for low-overhead exploratory probes, **not** an automatic replacement for a validated formal runner.
- **Local edit loop:** investigate warm Julia sessions (`Revise.jl`/`TestEnv.jl` or simpler) only where measured compilation/startup costs justify them. Formal comparisons still use fresh pinned processes.
- **Statistical analysis:** unify terminology, paired log-ratio arithmetic, confidence-bound/decision tests, metadata checks, and reports through small tested shared helpers or adapters where they genuinely reduce duplication. All three repositories currently have Python analysis scripts, so extraction MAY begin there rather than introducing a new Julia runtime dependency.
- **GitHub/reporting:** evaluate `Tachometer.jl` or `AirspeedVelocity.jl` for orchestration, PR comments, and history only after proving they can consume the valid existing measurements **without replacing calibrated inferential/accuracy rules**. Do not maintain both as parallel permanent pipelines without a distinct need.

**BC-31 — Evidence-based reuse.** Standardization SHOULD begin with commands, result interpretation, reports, and tested common statistical functions. The packages MAY keep different runner implementations. Creating a new shared Julia/Python benchmarking package is a separate decision requiring evidence of actual duplication, clear versioning, migration safety, and lower total maintenance cost.

**BC-32 — Stable user-facing interface (eventual).** A common **conceptual** interface SHOULD offer *list*, *explore*, *confirm*, and *report*, whether as thin local scripts or a shared executable. It is **not required** that the current `PB_*`, `PS_*`, and `HDM_*` settings be renamed; existing commands must keep working. Proposed (not yet implemented) example:

~~~text
bench list --group selected
bench explore --case CASE_ID
bench confirm --base BASE_SHA --candidate HEAD --primary cases.txt
bench report --experiment EXPERIMENT_DIR
~~~

No tool may label exploratory output confirmed merely because these commands share an executable.

## 10. Adoption order and conformance checks

This is a target contract with staged opt-in. Do not close issues or block release based solely on unimplemented future items.

| When | Scope | Owners / acceptance |
|:--|:--|:--|
| **Now** | Measure edit-to-feedback latency, test narrow selection, and establish a fast local pilot; retain existing working harnesses | ProximalSolvers #110 Stage 1 is the cross-package pilot; ProximalBase #514 targets reliable fixture selection. Demonstrate measured reductions **without** compromised certificates |
| **Before 0.6** | Preserve pinned pre-migration solver/estimator baselines and perform a small matched-accuracy upgrade comparison | ProximalSolvers #110, HDM #216, migration #527. Store raw evidence, explicitly review material changes |
| **After 0.6, unless specific need arises earlier** | Independently calibrate A/A variability and formal improvement/noninferiority decisions | ProximalBase #514/#462, ProximalSolvers #110, HDM #216. Tests include zero effect, synthetic rescaling, actual slowdowns, timer-limited/noisy cases, missed accuracy, and missing data |
| **After local/CI pilots stabilize** | Reuse targeted formal results in performance-sensitive PR review and, if justified, add history/reporting automation | Existing package CI ownership remains intact (#121, #195, #138); no noisy hard PR timing gate |
| **Optional later** | Extract shared analysis and/or orchestration code, with a compatibility adapter and one maintained implementation | Demonstrate parity with all three original analysis fixtures and less maintenance before removing local copies |

A package's adoption PR SHOULD include an explicit **rule-to-test matrix** for the BC IDs it claims to implement, machine/Julia/dependency context, locally rerunnable commands, before/after development latency or A/A evidence where applicable, and evidence that the previous harness's correctness and failure accounting did not regress.

### Minimum acceptance scenarios for a future shared implementation

1. **Explore is fast but honest:** a single selected case reproduces the same scenario/checks as full selection, and the tool does not claim confidence from one run.
2. **Sabotage fails:** mutate a supposedly read-only fixture or return an incorrect/nonconverged solution; reference/accuracy validation fails, preserving the failed observation.
3. **Environment mismatch is not green:** change the harness, a required setting, a dependency, or a source identity; formal comparison rejects it or calls it an explicitly different experiment.
4. **A/A is measured independently:** identical revisions on controlled hardware produce a published noise assessment; uncalibrated important cases are not auto-gated.
5. **Known effect is detectable where expected:** deliberate slowed and accelerated workloads produce logically correct uncertainty classifications; timer-limited cases are inconclusive.
6. **No optimistic selection:** repeatable independent confirmation of a candidate chosen from many exploratory attempts.
7. **Failures do not disappear:** missing cases, interrupted processes, certificate misses, and unexpected allocations survive to the final report.
8. **CI remains responsive:** an ordinary low-risk PR gets targeted correctness/integrity feedback without the full statistical timing campaign.
9. **Migration is not misattributed:** a ProximalBase 0.5-to-0.6 stack comparison declares changed dependencies and preserves the old baseline.
10. **Tool sharing is justified:** common helpers/automation reproduce existing package-specific evidence semantics without forcing unnecessary engine conversions.

## 11. Explicit open decisions for sign-off

The following are **not yet decided** by this draft and require package-specific or cross-package review before automated enforcement:

1. **Performance margins:** which solver/estimator workloads warrant 5%, 10%, a practical absolute-time target, or no automatic margin; no global solver-level 5% default is implied.
2. **Inferential strategy:** minimum process-pair count, whether and how sequential looks are justified, use of run-level means versus medians, and any family-level error control.
3. **Controlled hardware:** availability, isolation and CPU-frequency practices, A/A acceptance thresholds, and whether a given environment can support hard gates.
4. **Shared implementation location:** independent scripts versus tested shared Python helpers/adapters versus a separate package; who owns versioning and compatibility.
5. **PR policy:** which risk classifications require a performance receipt, who approves primary cases and intentional slowdowns, and when an inconclusive result may be explicitly waived.
6. **Dashboard/tool choice:** whether Tachometer, AirspeedVelocity, or existing scripts provide a net improvement after a representative pilot.

**Sign-off criterion:** agree on the shared requirements and open-decision owners, then adopt them per repository through small, tested PRs with measured performance/maintenance benefits. This document is the single **target specification**; package-specific implementation plans and tests are its companions, not competing policies.
