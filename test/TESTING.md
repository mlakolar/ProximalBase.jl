# Test tiers: baseline, placement and measurements

This file records why the test suite is split into tiers, where each file sits and why, which
tests moved between files, and what the tiers cost on GitHub-hosted runners.
[CONTRIBUTING.md](../CONTRIBUTING.md) is the user guide; [`tiers.toml`](tiers.toml) lists the
files of each tier and [`.github/tier-rules.toml`](../.github/tier-rules.toml) says which tier CI
runs for a change.

## Baseline (before tiers)

On 2026-11-01 every pull request ran the core tier (`--core`: every file except
`precision_mixed`, `large_inputs`, `spectral_overflow` and `conformance/*`) on Julia lts and
Julia 1, Ubuntu, 4 workers each; Julia 1 also collected coverage. Master ran all 135 files on
five jobs. Times from the job logs of run 41348877210 (PR) and 41341537016 (master):

| Run | Job | Files | Job time | Worker time (sum over files) |
|---|---|---:|---:|---:|
| PR, core | Julia lts, Ubuntu | 117 | 20.1 min | 4252 s |
| PR, core | Julia 1, Ubuntu, coverage | 117 | 29.2 min | 6306 s |
| master, all | Julia lts, Ubuntu | 135 | 30.3 min | 6518 s |
| master, all | Julia 1, Ubuntu, coverage | 135 | 33.0 min | 6968 s |
| master, all | Julia pre, Ubuntu | 135 | 37.1 min | 8013 s |
| master, all | Julia 1, Windows | 135 | 37.2 min | 7734 s |
| master, all | Julia 1, macOS (2 workers) | 135 | 46.0 min | 4458 s |

A PR therefore waited about 29 minutes for its slowest required job (other PR runs that day:
26–30 min). With 4 workers the wall time is bounded below by a quarter of the worker time, so
balancing the files better could not have helped much: the work itself had to shrink.

**Compilation dominates.** Every test file was run twice in one fresh process on the Discovery
cluster (AMD EPYC 7542, one file per core, `--check-bounds=yes`, commit 83a2eef9), first cold
and then again in a new sandbox module, which reuses the package methods compiled by the first
run:

| Julia | Cold, sum over 135 files | Compilation in the cold runs | Warm rerun | Execution (warm minus its compilation) |
|---|---:|---:|---:|---:|
| 1.10.12 | 5856 s | 5457 s (93%) | 892 s | 98 s (1.7%) |
| 1.13.3 | 6662 s | 6057 s (91%) | 1032 s | 102 s (1.5%) |

The warm reruns still compile the closures and types each test file defines. Executing the
assertions takes under 2% of the time: the cost is compiling specializations. ParallelTestRunner's
`--verbose` output, now on in CI, shows the same on GitHub runners: 94–98% compilation per tier
(Julia 1.13). A test file's cost grows with the distinct method
specializations it reaches (function types × element types × shapes × wrappers × iterate
types), not with its number of assertions. Coverage on Julia 1 made it worse: code under
coverage is compiled without the package image, which is why the Julia 1 PR job took 6306
worker seconds against 4252 on lts.

The most expensive files on the Julia 1 PR job (worker seconds, lts / Julia 1):
`gpu` 377 / 573, `conjugate_projections` 343 / 465, `constrained` 112 / 187,
`blockchecks_wrappers` 146 / 186, `symmetric_groups` 109 / 168, `blockchecks_steps` 128 / 162,
`metric` 103 / 155, `blockchecks_values` 81 / 139, `stepchecks` 68 / 137, `hyperslab` 84 / 129,
`blockchecks` 97 / 109, `smooth` 71 / 107. `allocations` (50 / 80) ran serially after the
parallel batch.

## How the tiers were chosen

- **Quick** (every PR) holds cheap files that each test a basic invariant across the public
  API: loading, Aqua, the generic interface and traits; iterates (active sets, value writes
  versus state copies, `dropzeros!`, symmetric coordinates and exact symmetry, aliasing,
  read-only handles); values, gradients and proximal operators of representative separable
  and nonseparable functions; conjugates as projections; constraints; spectral functions;
  precision, overflow and non-finite boundary cases; validation before writes; zero
  allocations on hot paths; and device arrays (JLArrays) with scalar indexing disallowed.
  The five most expensive files contributed representative cases, moved into new files
  (below), so that Quick keeps their main identities and regressions without their grids.
- **Standard** adds the files of most function families, wrappers, constraints, smooth losses
  and the examples: what an ordinary implementation change can break.
- **Thorough** adds what shared or numerical changes can break: precision and overflow
  recovery, non-finite values, spectral domains and tolerances, the conformance suite,
  symmetric writes, products and parameters, block steps, metrics, the wrapper calculus,
  the remaining iterate and constraint grids, and the specialized families (cones, graphs,
  fibers, polyhedra).
- **Full** adds the exhaustive catalogs whose cases repeat paths the tiers above already reach:
  the rest of `gpu`, `conjugate_projections` and `allocations`; `precision_mixed`; the block
  checks on steps, values, wrappers and workspaces; the frozen weight-step formulas; the
  complex-input rejection catalogs (`real_*`); the symmetric mutator and parameter inventories;
  separable block traversal; the wrapper rejected-step catalog (`stepchecks`); and the family
  files `hyperslab`, `image`, `distl2` and `precompose_array`.

Moving a file to a later tier never removes it from pre-merge testing of its own code: the
rules in `.github/tier-rules.toml` run a family's files beside any change to that family
(`focus`), for example `hyperslab` with `src/functions/sets.jl` and `stepchecks` with
`src/calculus.jl`, and every changed test file runs beside the tier.

Every file is listed once in `tiers.toml`; full runs all of them, so the full tier runs
exactly the files the old complete suite ran, plus the new files split out of them.
`test/tier_selection.jl` checks this, the cumulative inclusion, and the CI risk rules.

## Measured times on GitHub

End to end is from the run's creation to its last job (the gate); the test jobs run in
parallel after the plan job (about 20 s). Ubuntu runners for this public repository have 4
cores; the CPU model varies between runs (AMD EPYC 7763, 9V74, Intel Xeon 8370C, 8573C), and the
same files take up to 1.8 times longer on a slow one, so each tier was run three times. Julia 1
has no coverage below full, and is then often faster than Julia lts. A cold cache would add
package installation and precompilation of the dependencies (a few minutes once per cache key).

| Tier | Target | End to end, runs 1 / 2 / 3 | Test job, Julia lts | Test job, Julia 1 |
|---|---:|---|---|---|
| quick | 3–5 min | 3.0 / 4.6 / 4.9 min | 2.4 / 4.1 / 4.4 min | 2.4 / 3.4 / 3.2 min |
| standard | 8–10 min | 11.7 / 9.6 / 9.5 min | 11.2 / 6.9 / 7.9 min | 8.9 / 9.1 / 9.0 min |
| thorough | 15–20 min | 26.4 / 21.2 / 16.7 min | 25.1 / 20.3 / 15.8 min | 25.9 / 20.7 / 16.2 min |
| full (five jobs) | about 30 min | 41.0 / 34.9 min | 31.3 / 24.4 min | 31.4 / 34.4 min (coverage) |

Runs 1 used the first placement (quick 34 files, standard 78, thorough 135); runs 2 and 3 the
measured placement (quick 37, standard 64, thorough 119 and 118). After run 3, `nonfinite_inputs`
moved from quick back to thorough (about 50 worker seconds on a slow runner), leaving quick at 36
files. Full: macOS took 40.5 and 23.5 min, Windows 37.1 and 23.3, Julia pre 30.1 and 29.7.

At the final placement (commit fafb1189): quick took 4.5 min end to end (run 41369981986; Julia
lts 4.0 min on an EPYC 7763, Julia 1 2.7 min on an EPYC 9V45; the runner itself 2.9 and 2.0
min). Full took 43.0 min as this PR's run (41369978673: macOS 41.1, Windows 36.3, Julia pre 35.0,
Julia 1 with coverage 33.4, Julia lts 20.5) and 35.1 min as a manual run without inputs
(41369984297). Both full runs and the thorough runs checked the conformance coverage map and
uploaded the reports.
Inside the job, about 50–70 s go to checkout, the Julia cache, the test environment and
precompiling ProximalBase with bounds checking (about 25 s); the rest is the test runner.

**Before and after.** PR runs on 2026-11-01 took 26–30 min end to end (median 28.4 min over
eight runs, all on the core tier). An ordinary PR now waits 3–5 min (quick), one that changes a
function family 8–10 min (standard), and one that changes shared code 17–21 min (thorough). The
full suite after merge takes 25–41 min, as master's complete runs did before (21–46 min per job).

## Tests moved into quick-tier files

Each split moved whole testsets, or split one loop's parameter tuple into a representative
subset and its complement, so that the original file plus the new file run exactly the
assertions the original ran. No assertion, tolerance, input or expected error changed and
nothing was deleted. Assertion counts (from the runner) confirm it.

| New file (quick) | From (later tier) | What moved | Assertions: before = after + new |
|---|---|---|---|
| `sparse_iterate_basics` | `sparse_iterate` (thorough) | 17 whole testsets: construction, `dropzeros!`/nnz, `activate!`, active coordinates, converting writes, `fill!`/norm, copies and the conversion-failure regression (#416), stored zeros in `mul!` (Float64), the symmetric matrix (coordinates, mirrored writes, exact symmetry, `copyto!`), triple products (Float32, Float64), aliasing of views | 2897 = 2577 + 320 |
| `allocations_basics` | `allocations` (full) | Loop entries: block steps on a `SparseIterate` and with `NormL1`/`GroupNormL2`; `GroupNormL2` with distinct group layouts; `NormL1` on vectors, sparse, atom and symmetric iterates; `GroupNormL2` prox; `LeastSquaresLoss` gradients | 2910 = 2766 + 144 |
| `constrained_basics` | `constrained` (thorough) | `NormL1` and centered `SqrNormL2` on four boxes with the brute-force optimality check; coordinate steps on a box; `Scaled(NormL1)` wrappers; simplices, closed forms, the zero-sum lasso KKT check, errors, general normals with a box, Float32 cases; Float64 open-domain boxes | 4108 = 3526 + 581 (+1, see below) |
| `conjugate_basics` | `conjugate_projections` (full) | Seven direct conjugate families (clipping, check loss, orthant, l2 ball, group l2 ball, huge·tiny scale, shift) in Float64 and Float32; the Float64 Moreau identity with infinite weights; nested mixed sums and failure atomicity; output aliasing of parameters; biconjugates; Float64 l1-ball radius accuracy and group l1 balls | 265693 = 259835 + 5858 |
| `gpu_smoke` (with `gpu_prelude`, `gpu_suite_basics`) | `gpu` (full) | `NormL1` (plain and weighted) and the user bits type; `NormL2`, `IndSimplex`; `GroupNormL2` on host and device layouts; weighted norms at device overflow; least-squares and logistic losses with a design matrix; infinite weights; Moreau array steps | 21767 = 21469 + 298 |

`constrained`: one check in `ProximalBase.Testing.test_prox` runs only for random candidate
points inside the constraint set, so its count depends on the random draws (110–112 for the
moved block under seeds 1–12); the moved code changed which draws later testsets see.

Moving cases changes the global random stream for the testsets that follow them in
`sparse_iterate`, `constrained` and `allocations` (each test file starts from `Random.seed!(1)`).
Every affected testset passes; none depended on its position in the stream. In `gpu`, every
remaining testset starts in the same random state as before. `test/cuda/runtests.jl` now runs
`gpu_prelude`, `gpu_suite_basics` and then, after reseeding, `gpu_suite`, so the physical GPU
covers exactly what it covered before, with the same random inputs as `test/gpu.jl`.

**Coverage.** The full tier's Codecov upload (Julia 1, Ubuntu, run 41361146694 at dad59957)
covers 14160 of 14618 lines (96.86%), the same 14160 lines as master at the base commit 83a2eef9.
**Assertions.** At the same base, every full job passes exactly 116 more assertions than master's
run 41341879373 (Julia 1: 639818 against 639702; broken counts equal): the 117 of
`tier_selection` less the one random-draw-dependent check in `constrained`.

## JLArrays and CUDA

`gpu_smoke` (quick) and `gpu` (full) run on JLArrays, GPUArrays' reference backend on the CPU,
with scalar indexing disallowed: they catch code that indexes device arrays element by element
or does not adapt its parameters. They do not test a physical device. `test/cuda` runs the same
shared cases on an NVIDIA GPU by hand (see its README); no tier and no CI job runs it, and
`test/oracles` (JuMP and conic solvers) is likewise run by hand.

## CI scenarios

Checked on GitHub with draft PRs against this branch (#431–#437, closed unmerged), so that each
diff held only the scenario's change:

| Scenario | PR | Plan | Outcome | PR merge gate |
|---|---|---|---|---|
| a test file in quick changed (`test/lla.jl`) | #431 | quick | passed, 4.1 min | passed |
| label `test:standard` added | #431 | standard | the next label cancelled it | **failed** (cancelled) |
| label `test:full` added | #431 | full, five platform jobs | cancelled by hand | **failed** (cancelled) |
| a test file outside quick changed (`test/hyperslab.jl`) | #437 | quick and `hyperslab` | passed, 4.6 min | passed |
| family code without a test change (`src/functions/norms.jl`) | #432 | standard | passed, 8.4 min | passed |
| shared code (`src/utils.jl`) | #433 | thorough | passed, 27.1 min | passed |
| documentation only (`docs/src/index.md`) | #434 | none | no test jobs, 0.4 min | passed |
| a failing assertion in a quick file | #435 | quick | both test jobs failed | **failed** |
| a test file missing from `tiers.toml` | #436 | the plan job failed | no test jobs | **failed** |
| the runner, CI and manifest (this PR) | #430 | full, five platform jobs | passed, 41 min | passed |
| manual runs of quick, standard and thorough | — | as chosen | passed | (named "CI result") |

Every check ran on the PR's head commit; the plan, test and gate jobs print it. The gate reads
the conclusion of each expected test job of its own run, so a skipped, cancelled, missing or
failed job fails it. Pushes to master and the nightly run can only be observed after the merge.
These scenarios ran with the first gate, a job after the tests; the next section describes the
gate that replaced it and the scenarios that checked it.

## The merge gate, the tier lists and the report upload (review of #430)

Three review comments on #430 changed the gate, the rule for `tiers.toml` and the conformance
report upload. Scenario PRs #443–#453 checked them on GitHub, against `ci/gate-scenarios` (a
scratch copy of this branch at f76a24d8) with a temporary ruleset that required `PR merge gate`
on that branch only, as branch protection will on master; both were deleted afterwards.

**The gate starts with the run.** The first gate had `needs: [plan, test, analysis]`, and GitHub
creates a job's check only when the job is queued. A label on an unchanged commit started a new
run, and until its tests finished the commit showed only the earlier, green gate (#431). The
gate job now has no `needs`, so its check exists, pending, from the start of the run. It polls
this run's jobs through the API (`scripts/ci_gate.jl`, read-only token) and passes only when
**Required jobs** (the former gate check, still after the plan, the tests and the benchmark
controls) passed in its own attempt. A failed, cancelled, skipped, missing or duplicated
Required jobs fails it, as do a run for another commit or event, 15 minutes of API errors, every
other job finished for 90 s without a Required jobs, and 170 minutes without a decision.
`test/ci_gate.jl` checks these decisions on job states, including re-runs and API errors. The
issue for a failing master run is opened by its own job, the only one with `issues: write`,
which runs no repository code and is skipped on pull requests.

Branch protection counts the newest check of a name: on #443, with the ruleset in place, the
earlier green gate and the new pending one were both listed as required and the PR was BLOCKED
(rollup PENDING) until the new run passed.

| Scenario | PR (run) | Plan | What GitHub showed | Gate |
|---|---|---|---|---|
| quick | #443 (41384664046) | quick | gate in progress 3 s after the run was created; tests passed | passed, 5.2 min |
| label `test:full` on the passing commit | #443 (41385185212) | full, five jobs | the label request returned at 03:50:00.9, the new gate was listed (queued) and the PR unstable at 03:50:02.5; BLOCKED once the ruleset existed; CLEAN between 04:30:06 and 04:30:27, after the gate passed at 04:30:18 | passed, 40.3 min |
| the same, with the ruleset from the start; full fails on Windows | #453 | quick, then full | CLEAN after quick; label request 04:22:46.7–04:22:48.8, CLEAN at 04:22:48.7, BLOCKED at 04:22:50.6; the Windows job failed and the PR stayed BLOCKED | failed |
| label `test:thorough`, then removed | #444 | quick, thorough, quick | each label change listed a new pending gate within 6 s; thorough passed | passed (21.2 min for thorough) |
| superseded by a new commit | #450 | standard | the first run was cancelled with its gate | cancelled |
| the new run cancelled by hand | #450 | standard | gate cancelled, PR BLOCKED | cancelled |
| a test failing on Julia 1 only (lts passed) | #445, attempt 1 | quick | Required jobs: Julia 1 failure | failed |
| Re-run failed jobs | #445, attempt 2 | | only Julia 1 re-ran; the gate waited for the new Required jobs | passed |
| a re-run of the gate alone | #445, attempt 3 | | the attempt lists the passed Required jobs under its own number; gate passed in 16 s | passed |
| Re-run all jobs | #445, attempt 4 | | gate pending within 4 s; the old Required jobs was not listed | passed |
| Re-run job on one test job of a passed run | manual quick run 41390953488, attempt 2 | | the job and Required jobs re-ran; the gate did not and kept its result (a job without `needs` is no dependent) | unchanged |
| a test file missing from `tiers.toml` | #446 | the plan failed | | failed |
| no Windows job (stub workflow, no tests) | #448 | full | Required jobs: "Julia 1 - windows-latest - x64 - full: missing" | failed |
| no Required jobs (stub workflow, renamed) | #449 | full | gate failed 95 s after the last job | failed |
| an invalid workflow file (first push of #448 and #449) | — | no run | no gate check at all; the ruleset blocks a missing required check | none |
| documentation only | #447 | none | no test jobs | passed, 21 s |
| `tiers.toml` only: `utils` moved from quick to standard | #451 | full, five jobs | | passed, 42.0 min |
| a row removed from the coverage map | #452 | thorough | the map check on Julia lts failed | failed |
| this PR merged with master (#413 added `test/symmetric_sphere.jl`) | #430 (41384482255) | the plan failed on the unlisted file | | failed; the file is now in standard |

**What remains.** Between a label (or push) and GitHub creating the new run, the previous gate is
still the newest check: on #453 the PR was BLOCKED less than 4 s after the label request
started, and on #443 and #444 the new gate was listed 2–6 s after the label. No workflow can
close that window. **Re-run job** on a single job does not re-run the gate (it depends on no
job), so a failure in such a re-run of a passed run leaves the gate green; CONTRIBUTING asks
for Re-run all jobs instead. The gate job holds one Ubuntu runner for the length of its run (about 5
minutes for quick, 40 for full); runners are free for this public repository but count towards
its limit of concurrent jobs. It makes about three API requests a minute, well within the
token's 1000 an hour per repository.

**Time after the tests.** From the last required test job to the gate's result took 6–9 s with
the first gate (runs 41369981986, 41369978673, 41369984297) and 12–22 s now (quick 12, 14, 21,
22 s; thorough 15 s; full 15, 21 s): Required jobs takes about 5–8 s to start and finish, and the
gate polls every 20 s until it sees Required jobs or every other job finished, then every 5 s.
Quick took 4.6–5.3 min end to end in these scenarios (5.0, 4.6, 5.2 and 5.3 min, with up to eight
runs at a time) against 4.5 min for the final quick run of the first round.

**`tiers.toml` gets full.** It decides what every tier runs, so a valid edit can move a regression
test out of the tiers a PR runs and still pass `tier_selection`, which checks that each file is
listed once. Any change to it, including the line of a new test file, now plans full (#451);
ordinary test files keep quick and run beside it.

**Conformance report upload.** A full run's macOS job passed its tests and then failed on
`Failed to CreateArtifact: Unable to make request: ETIMEDOUT` (run 41373888409). Every job now
checks that the test step wrote the reports (required), retries a failed upload once after 30 s
(`overwrite: true`), and reports the outcome in its summary from the artifact id GitHub returned.
The Julia lts, Linux upload, from which the coverage map is regenerated, stays required; on the
other jobs a second failure is a warning. Two manual thorough runs on scenario branches whose
first upload pointed at a missing directory checked this: in 41385714788 the lts upload
succeeded on the retry (notice) and Julia 1 failed twice (warning, no artifact, job passed);
in 41385717877 the lts upload failed twice, which failed the job and the gate, and Julia 1
succeeded on the retry.

## A test made independent of its worker

`gpu_suite.jl`'s ambiguity check called `Test.detect_ambiguities(ProximalBase)`, which also sees
methods that other test files add to ProximalBase functions in the same worker process. With
`symmetric_parameters_trait` (its `MyL1` adds an `lla_weight` method) earlier on the worker, the
check failed: on Discovery with Julia 1.13.3 and no test history (files in random order), and
locally by running the two files in one process. Which files share a worker depends on the
tier and on the timing history, so the check now ignores methods defined in test modules. The
known package ambiguities and the assertion are unchanged. This is the only assertion whose
input changed in this work.

## Per-file times

From the full run 41364805347 (Julia lts on an AMD EPYC 9V74, Windows on an EPYC 9V45). A file's
time includes compiling what earlier files on its worker had not compiled, so it depends on
the order: `gpu_smoke` takes 9 s on Windows because `gpu` had already run on that worker.

<details><summary>Every test file, its tier and its time</summary>

| File | Tier | Julia lts, Ubuntu (s) | Julia 1, Windows (s) |
|---|---|---:|---:|
| `gpu_smoke` | quick | 51 | 9 |
| `conjugate_basics` | quick | 28 | 26 |
| `losses` | quick | 23 | 27 |
| `wrapper_buffers` | quick | 20 | 14 |
| `atom_iterate` | quick | 20 | 21 |
| `spectral` | quick | 19 | 14 |
| `constrained_basics` | quick | 19 | 18 |
| `aqua` | quick | 12 | 14 |
| `spectralfunction` | quick | 12 | 10 |
| `sparse_iterate_basics` | quick | 11 | 10 |
| `sets` | quick | 10 | 11 |
| `interface` | quick | 10 | 8 |
| `symmetric_products` | quick | 10 | 9 |
| `scaled_overflow` | quick | 10 | 7 |
| `allocations_basics` | quick | 10 | 3 |
| `entropy` | quick | 9 | 10 |
| `intersection` | quick | 8 | 9 |
| `utils` | quick | 8 | 5 |
| `nested_box_values` | quick | 8 | 7 |
| `deprecated` | quick | 6 | 6 |
| `matrixview` | quick | 6 | 4 |
| `active_handle` | quick | 5 | 2 |
| `lla` | quick | 5 | 4 |
| `caches` | quick | 5 | 5 |
| `block_step_failures` | quick | 4 | 8 |
| `iterate_queries` | quick | 3 | 3 |
| `spectral_mask` | quick | 3 | 4 |
| `tier_selection` | quick | 2 | 2 |
| `lapack` | quick | 2 | 2 |
| `symmetric_parameters` | quick | 1 | 4 |
| `conformance/tolerances` | quick | 1 | 1 |
| `readme` | quick | 1 | 1 |
| `model_precision` | quick | 1 | 1 |
| `spectral_box_tolerance` | quick | 1 | 2 |
| `docstrings` | quick | 0 | 0 |
| `compat` | quick | 0 | 0 |
| `smooth` | standard | 83 | 75 |
| `separable` | standard | 53 | 51 |
| `nonconvex` | standard | 50 | 46 |
| `distance` | standard | 49 | 40 |
| `constrained_nonconvex` | standard | 38 | 32 |
| `examples` | standard | 32 | 30 |
| `second_order` | standard | 31 | 27 |
| `ballconstrained` | standard | 29 | 28 |
| `norms` | standard | 28 | 30 |
| `singular` | standard | 28 | 26 |
| `sqrhinge` | standard | 24 | 21 |
| `sumlargest` | standard | 24 | 23 |
| `binary` | standard | 22 | 21 |
| `cubenorm` | standard | 22 | 18 |
| `minimum` | standard | 21 | 21 |
| `ordered` | standard | 21 | 20 |
| `composition` | standard | 20 | 21 |
| `metric_steps` | standard | 20 | 18 |
| `soc` | standard | 18 | 15 |
| `affine` | standard | 18 | 16 |
| `envelope` | standard | 18 | 16 |
| `infinite_weights` | standard | 18 | 15 |
| `stiefel` | standard | 18 | 13 |
| `matrixsets` | standard | 17 | 12 |
| `bernoulli` | standard | 15 | 15 |
| `sparse_value_cost` | standard | 14 | 12 |
| `sparsegroup` | standard | 11 | 19 |
| `sparse_iterate` | thorough | 128 | 64 |
| `symmetric_groups` | thorough | 118 | 104 |
| `metric` | thorough | 114 | 105 |
| `blockchecks` | thorough | 104 | 71 |
| `constrained` | thorough | 100 | 107 |
| `symmetric_writes` | thorough | 100 | 82 |
| `symmetric_argument` | thorough | 79 | 44 |
| `block_step` | thorough | 64 | 73 |
| `spectral_nonfinite_values` | thorough | 61 | 60 |
| `calculus` | thorough | 57 | 62 |
| `symmetric_mul` | thorough | 47 | 19 |
| `spectral_overflow_wrappers` | thorough | 42 | 30 |
| `spectral_overflow` | thorough | 40 | 41 |
| `graph` | thorough | 40 | 28 |
| `symmetric_parameters_paths` | thorough | 39 | 26 |
| `gradient_alias` | thorough | 39 | 33 |
| `symmetric_parameters_trait` | thorough | 36 | 25 |
| `extreme_normals` | thorough | 35 | 27 |
| `fibers` | thorough | 29 | 22 |
| `conformance/spectral` | thorough | 29 | 19 |
| `expcone` | thorough | 28 | 27 |
| `symmetric_parameters_writes` | thorough | 28 | 19 |
| `polyhedron` | thorough | 28 | 26 |
| `box_bounds_overflow` | thorough | 28 | 23 |
| `testing` | thorough | 28 | 31 |
| `infinite_group_weights` | thorough | 27 | 37 |
| `value_overflow_fused` | thorough | 25 | 25 |
| `converted_weights` | thorough | 25 | 25 |
| `spectral_overflow_prox` | thorough | 24 | 13 |
| `weighted_overflow` | thorough | 23 | 19 |
| `partial` | thorough | 23 | 16 |
| `precision` | thorough | 23 | 35 |
| `rotatedsoc` | thorough | 21 | 18 |
| `conformance/element_types` | thorough | 21 | 14 |
| `spectral_domain` | thorough | 19 | 12 |
| `conformance/harness` | thorough | 18 | 18 |
| `conformance/gpu_arrays` | thorough | 17 | 44 |
| `nonfinite_inputs` | thorough | 17 | 14 |
| `spectral_tolerance` | thorough | 16 | 20 |
| `symmetric_parameters_merge` | thorough | 15 | 13 |
| `symmetric_parameters_extensions` | thorough | 15 | 11 |
| `conformance/linear_sets` | thorough | 15 | 12 |
| `conformance/data_functions_2` | thorough | 15 | 10 |
| `large_inputs` | thorough | 13 | 15 |
| `conjugate_steps` | thorough | 11 | 9 |
| `conformance/special_values` | thorough | 10 | 12 |
| `value_overflow_sqrnorm` | thorough | 9 | 11 |
| `conformance/kernels` | thorough | 9 | 11 |
| `conformance/data_functions` | thorough | 9 | 8 |
| `conformance/steps` | thorough | 9 | 8 |
| `conformance/failures_storage` | thorough | 8 | 10 |
| `conformance/wrappers` | thorough | 6 | 8 |
| `conformance/linear_sets_2` | thorough | 5 | 5 |
| `conformance/second_order` | thorough | 4 | 4 |
| `logdet_overflow` | thorough | 2 | 4 |
| `gpu` | full | 390 | 355 |
| `conjugate_projections` | full | 359 | 299 |
| `precision_mixed` | full | 318 | 306 |
| `blockchecks_wrappers` | full | 147 | 100 |
| `blockchecks_steps` | full | 129 | 110 |
| `blockchecks_values` | full | 85 | 88 |
| `hyperslab` | full | 83 | 89 |
| `weight_steps` | full | 82 | 48 |
| `stepchecks` | full | 68 | 88 |
| `allocations` | full | 62 | 52 |
| `symmetric_parameters_centers` | full | 61 | 43 |
| `image` | full | 54 | 56 |
| `distl2` | full | 48 | 50 |
| `real_wrappers` | full | 44 | 50 |
| `real_functions` | full | 39 | 32 |
| `precompose_array` | full | 37 | 44 |
| `real_iterates` | full | 35 | 27 |
| `symmetric_mutators` | full | 33 | 38 |
| `separable_blocks` | full | 29 | 33 |
| `symmetric_parameters_inventory` | full | 28 | 27 |
| `symmetric_mutator_errors` | full | 24 | 36 |
| `blockchecks_workspace` | full | 20 | 26 |
| `symmetric_parameters_inventory_walk` | full | 1 | 1 |

</details>
