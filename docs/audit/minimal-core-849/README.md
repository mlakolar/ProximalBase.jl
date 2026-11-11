# ProximalBase minimal-core rebuild (#849)

**Status (2026-11-10 PT):** Phase A declaration/dependency audit accepted by the independent architecture critic; **no package source files modified or removed yet**. Extraction and the minimal implementation remain to be done.

The separate fresh GPT-6.1-Sol/high critic approved Phase A: “APPROVE Phase A
architecture/dependency audit and Phase B extraction based on latest 5,476
declarations / 1,306 selected.” The critic independently checked private-symbol
closure; four remaining references are the explicitly narrowed `_ArrayPrecompose`
union arms. Final-source load, ambiguity, correctness, docs and performance review
remain separate gates. Regeneration was byte-for-byte deterministic under Julia
1.13.3, coverage includes all 52 paths and all 14 retained public types, and
Phase A left the frozen source tree unchanged.

Read [extraction-boundaries.md](extraction-boundaries.md) first. The reproducible
Julia AST inventory covers all 52 frozen files: 5,476 declaration rows (including
literal generated-method instances), 23,781 conservative lexical dependency
edges, and 10,983 external symbol/path line-list records. The proposed disposition
is 407 retained storage declarations, 899 extracted core/helper declarations,
and 4,170 archived declarations. Counts describe the audit, not a runtime method
table or a correctness result.

Regenerate without loading package code or installing dependencies:

```sh
/path/to/julia --startup-file=no docs/audit/minimal-core-849/generate_inventory.jl
```

The generator reads frozen blobs using `git show`, never the active source tree.
[inventory_policy.jl](inventory_policy.jl) gives the declaration selection rules;
[declarations.tsv](declarations.tsv), [dependency-edges.tsv](dependency-edges.tsv)
and [symbol-targets.tsv](symbol-targets.tsv) hold the graph.
[declaration-summary.tsv](declaration-summary.tsv) covers every baseline path,
[source-segments.tsv](source-segments.tsv) records exact top-level byte spans,
[module-hooks.tsv](module-hooks.tsv) lists module/include/precompile hooks, and
[external-references.tsv](external-references.tsv) locates tests/docs/extensions/
benchmarks/scripts. Static lexical edges deliberately overapproximate runtime
dispatch. The boundary document identifies union narrowing and body edits needed
to remove deferred references; Phase B still requires load/test/ambiguity checks.

## Immutable *content* reference and recovery paths

- Full parent commit: `eff21b6d7d33c622313122f46c6ee71693c101ee` (#848 squash merge).
- Full tree: `c76123a0109c0a0f683cb75085371696dffe1c8c`.
- Exact remote-verified branches: `archive/full-before-reset-2026-11-10`, `archive/pre-rebuild-2026-11-10`, `legacy/full`. The branch `refactor/minimal-core` began at the same SHA and is **the only implementation target**.
- Every other open PR (#840, #801, #558, #535, #377, #348, #299, #155) has a separate `archive/inflight/pr-N-2026-11-10` branch at its exact original head. `diag/issue-747`, `research/843-oracle-interface-audit`, and `audit/pb848-block-updates-20261111` have additional archived refs.
- The annotated non-release tag `preservation/post-848-full-2026-11-10` has been created and remote-verified: tag object `5239e1043a370f999753710e958fb2742aa786ac`, peeled commit equal to the full parent above. The archive branches are currently **not protected**. Administrator action is still needed to protect archival refs; do not describe the branches as immutable. A checkout of the full commit itself remains recoverable by its exact SHA as long as refs exist.

The complete path-level snapshot of **52 tracked Julia source files, 1,649,968 bytes**, with exact blob hashes and provisional disposition is in [source-inventory.tsv](source-inventory.tsv). This is **not yet** the declaration/call dependency audit; do not blindly delete by filename.

## Target milestone 0: small, working module

1. **Keep all iterate storage types.** `SparseIterate`, `BlockIterate` (never `AtomIterate`), `SymmetricSparseIterate`, along with `coordinates`, `activecoordinates`, `activevalues` when available, `ncoordinates`, `nactive`, `isactive`, `activate!`, `dropzeros!`, `blockview`, `blockindices`, `matrixview`, scalar/array access and basic products. Preserve deliberate active zeros, aliases, copies, arbitrary ordered/noncontiguous block layouts, and symmetric independent-coordinate semantics.
2. **Keep the stable function interface, not every concrete function.** `AbstractFunction`, `SeparableFunction` where needed, relevant traits, callable `f(x)`, `gradient[!]`, `value_and_gradient[!]`, `prox[!]`, `value_and_prox!`, `prox_block[!]`, `value_entry`, basic `lipschitz`, and only the coherent additional hooks required by retained functions. The new `prox_block[!]` stays mathematical and does not own iterate storage. Do not resurrect `prox_coordinate!`, `prox_entry!`, or storage-aware proximal methods.
3. **Keep minimal squared/quadratic functions.** `LeastSquaresLoss`, `SqrNormL2`, `QuadraticFunction`, and `Squared` if the scalar-loss contract is retained. Support ordinary vector/multiresponse forms of least squares when inexpensive and already tested.
4. **Keep tiny lasso dependencies for PS #234.** `NormL1` and `Scaled`, permitting `(1/n) * LeastSquaresLoss(y,X)`, plus small `ZeroFunction`/`LinearFunction` support if self-contained. The resulting PB core must support the mathematical building blocks of a lasso; actual solvers remain in PS.

Defer full GLM families, group penalties/group-lasso math, constrained sets/oracles, nonconvex/ordered/spectral/nuclear operators, general composition wrappers, LMO/Frank–Wolfe, Newton machinery, GPU implementations, and historical test catalogs for removed features. Do **not** remove generic block storage just because group penalties are deferred.

## Work sequence for the implementing agent

**A. Audit before deletion.** Use the frozen tree plus [source-inventory.tsv](source-inventory.tsv). Expand every source file into a declaration-level inventory: types, constructors, dispatches/typed signatures, method dependencies, module include order, exports, extensions, test entry points, docs, numerical contracts and precompile calls. Classify each retained declaration or segment as *keep unchanged*, *extract*, or *defer*. Resolve dependency cycles before touching source and publish the full audit in this PR.

**B. Extract minimal types and functions in narrow commits.** `src/functions/smooth.jl`, `functions/separable.jl`, `functions/glm.jl`, `calculus.jl`, `utils.jl`, `model_data.jl`, `metric.jl`, and `symmetric_parameters.jl` are mixed-purpose; take only needed declarations rather than retaining whole source families or changing numerical formulas. Preserve Float32/Float64 corner cases, NaN/Inf policy, source alias rules, zero behavior and precision. Do not add compatibility shims solely to keep retired names.

**C. Shrink the actual module.** Update `src/ProximalBase.jl` exports, imports and include order; remove deferred source files from the *active minimal branch* and exclude their test/extension/docs/precompile hooks. Do not leave dangling exports or typed references to undefined types. Keep all removed material on `legacy/full` and the archive branches. Record each deletion and rationale.

**D. Prove the minimal package works.** Fast package load, independent mathematical references for the retained functions, basic construction/copy/alias/active-zero tests for all three iterates, Float32/Float64, scalar/block/array proximal consistency, least-squares gradient/value accuracy, `Scaled`/lasso example, docs build and minimal extension-author example. Benchmark retained iterate hot paths against **post-#848 frozen source** with a compatible same-machine/same-harness A/B and A/A control; keep historical `def81ec9` as a separate diagnostic reference. Attribute any >5% hot-path slowdown explicitly. No claim of a full historical package-suite pass.

**E. CI and integration.** For this rebuild, gate PRs with **Linux on current Julia 1.x only**. No Julia LTS, Windows/macOS or GPU jobs are necessary on the minimal-core PR. Adapt existing CI inventory/tier logic to the reduced, actually supported tests; do not hide failures for retained functionality. Keep old full-suite resources on `legacy/full`. No ProximalSolvers source/pin changes; PS #234 migrates separately after PB stabilizes. No registration or 0.6 release merely because the reduction merges.

## Review / merge requirements

Open one ordinary draft PR targeting protected `master`. Require a declaration/dependency audit before accepting code deletion, a meaningful correctness and performance comparison, final-SHA Linux/current-Julia CI green, independent final-source review, and a normal protected-branch merge. Keep source modifications restricted to `refactor/minimal-core`; no force pushes or history rewrite.

**Not yet completed:** archival-ref protection; extraction/removal; minimal tests and CI; final independent source review; merge. See [umbrella #849](https://github.com/mlakolar/ProximalBase.jl/issues/849) for the authoritative feature scope.
