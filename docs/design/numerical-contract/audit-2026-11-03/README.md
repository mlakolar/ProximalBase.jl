# Audit of closed issues and pull requests, 2026-11-03 (#556)

This folder records how [`known-limitations.md`](../known-limitations.md) was
built: every closed issue and pull request of ProximalBase.jl was read for
limitations it reported, and each one was checked against the current
`master`. The register holds the current facts; this folder holds the trail.

## Baseline

| | |
|:---|:---|
| `master` checked | [`920ebece7cea104eba61bb1e9c751c347444c802`](https://github.com/mlakolar/ProximalBase.jl/tree/920ebece7cea104eba61bb1e9c751c347444c802) (0.6.0-DEV, #543 merged) |
| GitHub data fetched | 2026-11-04T06:00:04Z (2026-11-03 local) |
| Julia | 1.13.3 and 1.10.12 (the minimum supported), macOS on Apple M3 Pro (aarch64) |
| devices | JLArrays 0.4.2 (CPU-backed GPUArrays) on Julia 1.13 only; no CUDA, ROCm or Metal hardware |
| full test suite at the baseline | Julia 1.13.3: 879,763 pass, 271 broken, 0 fail (27 min); Julia 1.10.12: 879,374 pass, 287 broken, 0 fail (22 min). The 16 extra broken on 1.10 are the allocation checks skipped on Julia < 1.11 |

## Counts

| | count |
|:---|---:|
| closed issues examined | 163 (160 completed, 3 not planned) |
| closed pull requests examined | 266 (231 merged, 35 closed unmerged) |
| discussion read | 958 conversation comments, 166 submitted reviews with a body, 686 inline review comments in 350 review threads (resolved ones included) |
| items that reported at least one residual limitation | 322 of 429 |
| candidate limitations extracted | 749 |
| candidates merged as duplicates within a theme | 252 (749 → 497 behaviors) |
| confirmed behaviors merged into register entries | 122 of 123 (112 from the history, 11 found while verifying them) → 80 entries; the 123rd is an allocation cost (#381), kept in the index |

Final classification, by behavior (497 from the history, plus 17 found
while verifying them or from the #556 seed list):

| classification | behaviors | candidates |
|:---|---:|---:|
| fixed (verified on `master`) | 148 | 254 |
| confirmed current limitation | 123 | 145 |
| conforms (not a limitation under the current contract) | 105 | 126 |
| intentionally unsupported | 53 | 81 |
| unverified | 53 | 77 |
| approved limitation | 15 | 27 |
| obsolete | 7 | 8 |
| duplicate | 4 | 4 |
| out of scope (infrastructure, process) | 6 | 27 |

The register's active sections hold 80 contract gaps (122 confirmed
behaviors), 7 approved limitations (8 behaviors) and 21 unverified entries
(26 behaviors). The other 27 unverified behaviors are performance claims not
re-measured, conformance-suite aggregates, downstream behavior or maintenance
risks; they are in `clusters.tsv` with their owners, not in the register. The
other 7 approved behaviors are accepted performance costs recorded on their
pull requests (#330, #411, #504, #512, #526, #529, #545); their sign-offs are
review comments on the maintainer's account followed by the maintainer's
merge.

## Files

| file | content |
|:---|:---|
| `items.tsv` | one row per closed issue and pull request (429): number, kind, title, URL, closed date, closing reason (merged, closed unmerged, completed, not planned, and the closing PR or commit), related items, whether it reported a residual, fixes named in its discussion, what it did, the audit disposition, and its behavior clusters |
| `candidates.tsv` | one row per extracted candidate (749): its first source (comment or review URL), category, status in its own thread, cluster, final classification and register ID |
| `clusters.tsv` | one row per behavior (514): classification, register ID, title, contract rule, owner, source items, candidates, tests and fixes cited, and the reason for the classification |
| `reproduce-active.jl` | the reproducer of every active register entry; prints `PRESENT`, `ABSENT` or `ERROR` |
| `method/` | the fetch commands (`fetch.sh`), the GraphQL and bundling scripts, the two briefs the extraction and verification agents followed, and `build_index.py`, which wrote the three TSV files |

## Method

1. **Baseline.** A fresh worktree of `origin/master` at the SHA above; the
   contract (`numerics.md`), `decisions.md`, this folder's README,
   `coverage.md`, `impact-on-open-issues.md` and issues #366, #486, #495, #556
   were read first.
2. **Inventory.** `method/fetch.sh` lists closed pull requests and closed
   issues through the paginated REST API (`--paginate --slurp`), removes the
   pull requests from the issue list, and fetches every conversation comment
   and inline review comment of the repository. Batched GraphQL adds
   submitted reviews, review-thread resolution, closing references,
   cross-references and close events; no connection reached its page limit.
   `bundle.py` writes one chronological bundle per item (4.6 million
   characters in all).
3. **Extraction.** Fifteen extraction agents each read about 330,000
   characters of bundles in full, following `method/extraction-brief.md`.
   For each item they recorded what it did and whether it reported a
   residual; for each candidate, the exact source comment, a short exact
   quote, the affected API, the behavior, the conditions, the types and
   backend, the expectation at the time, follow-up issues and pull requests,
   and the matching current contract rule. Every item number appears exactly
   once, and the quotes were checked as substrings of the bundles.
4. **Verification.** Seven verification agents, one per theme, followed
   `method/verification-brief.md`: merge duplicates, find the current code,
   search the later history for fixes, read the current tests and docs,
   reproduce on `master` (recording any change to the old reproducer), run
   confirmed cases on both Julia versions, and classify against the current
   contract with one rubric. The seed cases of #556 were reproduced by the
   coordinator first (`s1_seed.jl`, both versions).
5. **Consolidation and independent recheck.** The coordinator merged
   duplicates across themes into register entries, chose one owner per entry
   (the most specific open issue, otherwise a #366 group tracker), checked a
   random sample of fixed classifications, and wrote `reproduce-active.jl`.
   Its run at the baseline prints `PRESENT` for all 79 runnable checks on
   Julia 1.13 (with JLArrays) and all 75 runnable ones on 1.10 (device checks
   skipped). LIM-072 rests on four `@test_broken` rows that are still broken in the
   full test run.

## The seed cases of #556

| seed | result on `master` |
|:---|:---|
| #216 / #543: `Conjugate` finite-result failures in later blocks | **Not confined to invalid or endpoint steps.** At valid positive steps (`Conjugate(SqrNormL2())` at `[1e300]`, `γ = 1e-10`; `γ = 1e-310`) the check raises `ArgumentError` and a block sum has already written block 1: LIM-001 (owner #216). The underlying failure is LIM-069 (#255). At invalid steps: LIM-002 (#311). |
| #323: search-discovered infeasibility | Reproduces for the three #323 cases and for an array normal: LIM-003. At a NaN argument or `γ = Inf` the empty set returns NaN with no error: LIM-004. |
| #311: value-dependent step rejections | `BernoulliLoss` and `CubeNormL2` reject negative steps only for some values, with `DomainError`, after block 1 is written: LIM-014. Most leaves accept negative and NaN steps silently: LIM-013. Unnamed endpoints give NaN: LIM-015. |
| #239: non-proximable later block | `MethodError` after block 1 is written: LIM-005 (owner #216). The other #239 rows are fixed (`Conjugate(ElasticNet)` value, nested boxes, `NuclearNorm` with `Float16`, all raise before writing) or obsolete (`IndHyperslab` at NaN now returns NaN under #287's rule). #239's remaining scope (#274) is closed. |
| #371: tuple `Conjugate` conversion | Both reproducers still write block 1 before `InexactError`: LIM-006. |
| #255: `Conjugate` at extreme steps | Still raises at valid steps for halfspaces, hyperplanes, mixed `SeparableSum`, spectral conjugates and `Float32` calls with `Float64` steps; `prox_entry` gives `-Inf` for a scaled box cone; array steps of `1e-310` are rejected (LIM-069). `Conjugate(IndBallL1)` and `Conjugate(IndSimplex)` at `[0, 1]` pass only because #529 maps one infinite entry of `x/γ` to its limit; at `[0.5, 1]` they still raise. |
| #543's unchecked scope (devices, `Float16`/`BigFloat` spectral entries) | The new preflights reject before writing on JLArray blocks, and `Spectral`, `SingularValue`, `NuclearNorm` reject `Float16` and `BigFloat` before writing (cluster P0-02). Real hardware is unverified: LIM-088. |

## Contradictions found in the current code and docs

- **Contract text.** The O6 block-sum paragraph added by #543 has no
  **Decided** line and conflicts with decision 42 and S4 (register, Q1). The
  `PrecisionError` docstring says no function raises it, while O6.3, A3 and
  decisions 18 and 42 send uncertifiable and unrepresentable results to it;
  the `Conjugate` check raises `ArgumentError` instead.
- **Examples on the contract page that `master` does not meet:** V10
  (LIM-039), T3's `QuadraticFunction` gradient (LIM-038), R1's `Translate`
  homogeneity example (LIM-023), P3's `[1e-50, 2e-50]` metric (LIM-016), E2's
  `NormL2` example (LIM-047), and T6's `Float16` rejection (LIM-045).
- **Docstrings that promise what the code does not do:** the spectral
  docstring says `Float16`/`BigFloat` arguments raise (LIM-046); the
  `Spectral`/`SingularValue` docstrings state that the overflow rule "does
  not change the standalone value" (LIM-030), and the `Conjugate` docstring
  documents Moreau cancellation as accepted behavior (LIM-070), both of which
  S4 does not allow; `iterates.md` says a complex operand of an iterate
  product raises (LIM-012). In the other direction, `interface.md` (GPU
  arrays) understates what runs on devices: `Conjugate` of the host-decomposed
  functions and separable array steps work on JLArrays.
- **Partial fixes:** #200 fixed the narrow-output coefficient for the slab
  only (LIM-020); #304's infeasibility pre-check covers scalar normals and
  scalar boxes only (LIM-003); #507 fixed the ridge slope helper but not the
  spurious infeasibility (LIM-019); #504/#526 left the symmetric iterate out
  of FN-9 scaling (LIM-073); the #255 example passes for an incidental reason
  (above).
- **Tests that pin behavior the contract has since changed:**
  `test/spectral_nonfinite_values.jl:36`, `test/infinite_weights.jl:58`,
  `test/threshold_factor_type.jl:144`, `test/separable.jl:413` (pre-decision-20
  NaN behavior) and `test/gpu_suite.jl:1746` (the cancelling device result of
  LIM-061). Skipped tests: the three `VERSION < v"1.11"` allocation skips in
  `test/conjugate_projections.jl` are stale (all 16 assertions allocate 0
  bytes on 1.10.12). The only `@test_broken` rows outside the conformance
  suite are `test/sparse_value_cost.jl:162-173` (LIM-072) and
  `test/matrixview.jl:252` (LIM-071).
- **Wrappers that behave differently from their leaf:** LIM-064 (`Constrained`
  around a value-less function), LIM-057 (`Scaled(indicator, Inf)`), LIM-051
  (`Precompose` around an indicator), LIM-050 (array-step route of the ℓ1
  ball and simplex). Also seen, not entered: `Constrained(SqrNormL2(1.0,
  ones(1, 2)), IndBox(0, 1))` accepts a center of the wrong shape in `prox!`
  but not in `value_and_prox!` (related to LIM-065, #312).
- **Gaps with no specific owner** (owned for now by a #366 group tracker):
  LIM-007, LIM-008, LIM-012, LIM-021, LIM-036, LIM-037, LIM-043, LIM-044,
  LIM-047 to LIM-049, LIM-052 to LIM-054, LIM-058. LIM-048 (NaN turned into
  finite values by `NormL0`, `IndBallL0`, `LogPenalty`, `CheckLoss`,
  `HingeLoss`, `SLOPE`, `NormL1Fused`, `IndIsotonic`) is not listed in #366.
- **Stale or closable issues:** #307 reproduces as fixed; #239's remaining
  scope is closed; #313 can only be triggered with `Float16` data, which T6
  excludes.
- **Smaller items:** the message at `src/calculus.jl:915` is cut off
  mid-sentence; an array step is printed as "γ = 2-element Vector{Float64}" in
  the `Conjugate` finite-result message; `Constrained(NormL0(1),
  IndSimplex(1))` names the inner wrapper instead of `NormL0`;
  `docs/design/symmetric-sparse-iterate-0.6.md` still opens with "Nothing here
  is implemented" next to its sign-off note; `coverage.md` was generated at
  `87825cd1` and does not reflect later fixes (left unchanged here, since
  this audit changes no conformance rows).

## Limits of this audit

- Extraction and verification were done by agents following the briefs in
  `method/`; the coordinator re-ran every active reproducer and spot-checked
  the fixed classifications, but not every one of the 148 fixed behaviors was
  re-derived by hand. A fixed classification rests on a current probe or a
  named passing test, and the full suite passed on both Julia versions.
- No GPU hardware: device results are JLArrays results (LIM-088).
- Performance claims were not re-measured on controlled hardware; they are
  listed in `clusters.tsv` as unverified, approved (with the sign-off), or
  fixed, with owners.
- Some classifications depend on reading the contract where it is silent;
  those are marked unverified with the clause in question, not confirmed.

## How to repeat

```sh
sh docs/design/numerical-contract/audit-2026-11-03/method/fetch.sh <empty cache dir>   # inventory and bundles
julia --project=. docs/design/numerical-contract/audit-2026-11-03/reproduce-active.jl   # recheck active entries
```

A later audit can extract from the new items only (compare `items.tsv` with
a new inventory), re-run `reproduce-active.jl` against the new `master`, and
move every entry that prints `ABSENT` to "Resolved since the audit" after
confirming the fix.
