# Wide least-squares solves: cached thin SVD against cached Cholesky (#1103)

## Question

For dense wide least-squares problems (`n < p`), how does the cached thin SVD that `LeastSquaresLoss` uses for its
proximal operator compare with a cached Cholesky solve in observation space, at the same accuracy? The comparison covers
first use (construction, factorization and first prox), the warmed prox, a change of the step, whole ADMM lasso runs,
warmed allocations, and the accuracy of each prox from the original data, including the reproducer of
[#1078](https://github.com/mlakolar/ProximalBase.jl/issues/1078). It decides whether a production change is warranted;
it does not make one, and it does not change `src/` or `test/`.

This folder holds the tooling and, once the campaign has run, its results. The plan, with the reasoning behind each
choice, is the comment
[on PR 1135](https://github.com/mlakolar/ProximalBase.jl/pull/1135#issuecomment-7058822422). This README states the
method as the scripts implement it and refers to the plan for the motivation.

Measured package source: `src/` of commit `fdf9b52777b280750a4402776ffeb94b7c702be6`. The scripts load `src/` of the
checkout that holds them unless `PB_SOURCE` names the root of another tree; a campaign points `PB_SOURCE` at a checkout
of that commit and passes the commit in `PB_SOURCE_SHA`.

## Files

| File | Role |
|---|---|
| `generate.jl` | writes the inputs as little-endian binary arrays with a manifest `inputs-index.tsv` (case, element type, design, response, shape, SHA-256 of each array, largest singular value of `X`) |
| `harness.jl` | the paired timing harness, the ADMM runs and the dump of the prox outputs |
| `accuracy.jl` | the BigFloat evaluation of each dumped prox output from the stored inputs |
| `summarize.py` | rebuilds the tables and applies the criteria from the raw rows (Python standard library only) |
| `common.jl` | shared: the package modules, the prototypes, the roles, the manifest reader, the ADMM loop |

The scripts use Julia standard libraries and run with `--project=.` at the package root. Raw output goes to
directories outside the repository; no data or output is committed.

## Roles

All roles are loaded in one process. Prototype modules and the package source are included twice, into modules
with different names, so that the A/A copies are compiled separately.

All three prototypes use `ρ = inv(T(γ))` as `src/functions/leastsquares.jl` does, with `T` the element type of `X` and
`y`, and form the shift `I + XX'/ρ` with the scalar `inv(ρ)` computed once, as `lasso.m` forms `1/ρ`. They differ in what
they cache and in the correction:

- **S**: the current `LeastSquaresLoss` prox. **S'**: a second include of the same source (its A/A control).
- **C_B**: the fat branch of Boyd's `lasso.m`. It caches `X'y` and the Cholesky factor of `I + XX'/ρ` and computes
  `q = X'y + ρv`, `β = q/ρ - X'((I + XX'/ρ) \ Xq)/ρ²`.
- **C_D**: the residual form. It caches the factor of `I + XX'/ρ` and computes `a = (I + XX'/ρ) \ (y - Xv)`,
  `β = v + X'a/ρ`; with `β = v + X'a'`, `(X'X + ρI)β = X'y + ρv` reduces to `X'((XX' + ρI)a' - (y - Xv)) = 0`.
- **C_W**: the whitened form. With `LL' = I + XX'/ρ`, `W = L⁻¹X` and `c = L⁻¹y` it computes `β = v + W'(c - Wv)/ρ`.
- **C_B', C_D', C_W'**: second copies of each prototype module (their A/A controls).

The prototypes check `all(isfinite, X)` with the error text of the SVD path, and check that the step converts to a
finite positive number of type `T` with a finite positive reciprocal. Setup caches `XX'`; `refactor!(c, γ)` re-shifts
and refactors from the cached `XX'` (C_W also recomputes `W` and `c`). The solves use BLAS and LAPACK calls on
preallocated buffers, and the 4-argument `prox!` checks that its step converts to the step of the current
factorization. For Float32 or Float64 data and a step of the same type, the warmed `prox!` of a factorized object
allocated no bytes in the probes of `harness.jl` (prox and refactor-and-prox, both element types).

The setup that is timed is the plain constructor. `breakdown` runs the same kernels with `time_ns()` readings between
them; `PB_CHECK_ONLY` verifies that the object it builds gives the same prox output bit for bit.

The prototypes are not tuned: a production version could fuse vector passes. The ratios belong to these prototypes.

## Inputs

`generate.jl` writes dense Float64 and Float32 cases with `n < p`, and `X` and `y` of the same element type.

- Aspect ratios `n × p`: 100×1000, 500×5000, 1000×2000, 1000×1100 (near-square), 200×20000, 1500×5000.
- Designs, each with unit-norm columns: Gaussian; rank `n/2` (a product of Gaussian factors); duplicated rows (the
  first `n/2` rows repeated, so that the rows are exactly equal and `XX'` is singular in exact arithmetic). The
  rank-deficient designs have a consistent response `y = Xβ*` and an inconsistent one `y = Xβ* + noise`; the Gaussian
  design has `y = Xβ* + noise`. For the product design, the stored `X` and `y = Xβ*` are computed in rounded arithmetic,
  so the rank and the consistency hold to the tolerances of the validator in `generate.jl`, not exactly. `β*` has a
  nonzero `N(0, 1)` entry with probability `min(1, 100/p)` (about 100 nonzeros for `p ≥ 100`) and the noise has
  variance `1e-3`, as in the wide lasso of Boyd et al., §11.1.
- Steps: `γ ∈ {1e-3, 1, 1e3, 1e6}` for Float64 and `{1e-3, 1, 1e3}` for Float32; the nominal penalties `ρ = 1/γ` (before
  rounding to the element type) are `1e3, 1, 1e-3` and, for Float64 only, `1e-6`. The lasso instances are also
  evaluated for accuracy at `γ = 0.1` and `10` (`ρ = 10` and `0.1`, the ADMM penalties); the manifest column
  `acc_gammas` lists the evaluated steps and `gammas` the timed ones. The manifest stores the largest singular value
  `σ_max` of each stored `X`, and the summary reports `ρ/σ_max²` for each case and step.
- Proximal inputs: eight columns per case. The first four are random `N(0, 1)` vectors and the last four are `z - u`
  iterates from an S-ADMM run on the same data (`ρ = 1`, `λ = 0.1‖X'y‖∞`) after 1, 5, 20 and 100 iterations (inputs a
  solver reaches). The generator checks that the four vectors of each kind are pairwise different, and the harness
  checks it again when it reads a case.
- The wide lasso instances of §11.1 are the Float64 Gaussian cases 1500×5000, 500×5000 and 200×20000. The timing groups
  use the Gaussian cases and the consistent responses of the rank-deficient designs; their inconsistent responses are
  used for the accuracy evaluation.
- Accuracy controls: the reproducer of #1078 (`X = [1 0]`, `y = [1]`, `v = [±1/ρ, 0]`, `ρ = eps(T)/8`) and neighbours:
  `ρ ∈ {eps/8, eps, 32 eps}`, both signs, null-space components in `v`, 1×3 and 2×4 coordinate and diagonal designs
  (one with a zero singular direction), rotated 1×2 and 2×4 designs (full and deficient rank), a square and a tall
  design, in Float64 and Float32. They are deterministic, they are used for the accuracy evaluation and not for timing,
  and the harness can generate their outputs alone (`PB_FILTER=c1078 PB_GROUP=outputs`).

## Integrity and provenance

- Every binary array is read from the bytes that were hashed: the file is read once, its SHA-256 is compared with the
  manifest, and the array is decoded from the same buffer. A mismatch or a size that differs from the manifest stops the
  process. The prox outputs are read the same way by `accuracy.jl`.
- Each tab-separated artifact ends with the line `# complete: yes` (a reference file with `complete yes`); a reader
  rejects a file without it, so an interrupted write is not mistaken for a result. `summarize.py` leaves such a file
  out and marks the verdict provisional.
- Each artifact carries the identities of the run: `source_sha` (the package source), `harness_sha` (this folder) and
  `inputs_index_sha256` (the manifest), and the timing files also the Julia version, the BLAS configuration and the
  OpenBLAS version, the CPU vendor and model, the L3 size, the load average and the thread counts. The two SHAs are
  labels that the caller passes in `PB_SOURCE_SHA` and `PB_HARNESS_SHA` (7 to 40 lowercase hex digits; they are required
  outside `PB_CHECK_ONLY`, and a copy without `.git` cannot check them against the files). The generator, the harness
  and the evaluator write them to the manifest, the timing, outputs, untimed and reference files and the accuracy files.
- The evaluator refuses outputs made from another manifest (before scoring anything), and checks the element type, the
  steps, the number of inputs and the byte count of each output file against the manifest and the identity of its
  harness against `PB_HARNESS_SHA`. The ADMM stage refuses untimed counts made from another manifest, instance, Julia
  version or source. `summarize.py` accepts an accuracy, outputs, untimed or reference file only when its manifest digest
  equals that of `inputs-index.tsv` and its source and harness identity equals that of the timing files.

## Measurements

One process writes one tab-separated file `timings-<julia>-<order>-<proc>-<groups>.tsv`: a header of `# key: value` lines,
one row per (case, measurement, parameter, round, position, role) with the time per call, the GC time, the bytes and the
allocations per call, and the closing line. A failure of one sample is recorded as a row `error` (the cell and the
message) and the process continues with the other samples; the summarizer treats the cell as incomplete. A warm-up for
each element type calls the routines of each role (the ADMM routines for Float64 only) before the first timed sample;
first use is therefore timed in a warmed process, with the state of the native libraries shared.

**Pairing.** In the warm and penalty groups, within a (case, cell), all roles run batches with the same number of calls,
from the same state, along the same sequence of steps and inputs, and take the minimum over the same number of batches
(`PB_BATCHES`, 3). A batch is a
set of complete cycles: a multiple of the number of inputs (4) and, for the penalty groups, of the length of the step
schedule, restarted at the first input and the first step. `PB_ORDER` (`S_first` or `C_first`) fixes the order of the
roles in the odd rounds (or ADMM repeats) of a process and the order is reversed in the even ones; within a round or
repeat the order is fixed. The calls per batch come from one calibration per case on a complete cycle (at `γ = 1` for
the warmed prox, on the schedule for the penalty groups): the smallest multiple of the cycle with at least 30 calls and
about 20 ms of calls (warm) or about 50 ms (penalty) for the slowest primary role, at most 3000 calls, and one cycle
with one batch when that role needs 0.1 s or more per call. The calibration is of that role at those settings only; the
rows `batch_plan` record the result and the slowest call time.

1. **First use** (`first_use`): construction, factorization and first prox on a fresh object at `γ = 1`, one sample per
   role and round, the input rotating through the four random vectors by round (the same for all roles of a round),
   `GC.gc()` before each sample outside the timed region, with GC time, bytes and allocations from `@timed`. The seconds of
   the setup steps of the primary roles are recorded too (`first_use_part`), on a separately sampled path:
   for S, a reconstruction of the cached SVD that omits the loss constructor (the copy with the finiteness check,
   `svd`, the materialized `V`, and `s²`, `U'y` scaled by the singular values and the allocation of the work vector);
   for the C roles, the kernels of the constructor (the finiteness check and allocation, `XX'`, `X'y` for C_B, the
   factorization, and for C_W the copy of `X`, `trsm` and the triangular solve of `y`).
2. **Warmed prox** (`warm`): for each timing step and each kind of input (the four random vectors, the four ADMM
   iterates), the minimum over the batches of complete input cycles, per role and round.
3. **Penalty change** (`penalty`): S calls with the step of the schedule; each C role refactors from the cached `XX'` at
   that step and then calls the prox. Every role starts each batch from the last step of the schedule, and no two
   consecutive calls (also across batches) use the same step. Two schedules, both of registered steps: alternating
   between `γ = 1` and `1e3` (`alt2`), and cycling through `1e-3, 1e3, 1, 1e6` for Float64 and `1e-3, 1e3, 1` for
   Float32 (`newstep`).
4. **Warmed allocations** (`alloc`, timed cases only): `warmed_bytes(op::F) where {F} = (op(); op(); @allocated op())`
   behind a function barrier, for the prox cycle and for the refactor-and-prox cycle, per call, for each role, with the
   reachable bytes of each object.
5. **Complete ADMM runs** (`admm_reference`, `admm_untimed`, `admm`), Float64, on the three lasso instances at
   `ρ ∈ {0.1, 1, 10}`. One loop serves all roles; only the x-update differs: `prox!(x, obj, z - u, 1/ρ)`. It follows
   Boyd's lasso loop with `α = 1`, `λ = 0.1‖X'y‖∞`, `z = soft(x + u, λ/ρ)`, `u += x - z` and the stopping rule with
   `ε_abs = 1e-4`, `ε_rel = 1e-2`.
   - `admm_reference`: S-ADMM at `ρ = 1` to the lasso stationarity `‖z - soft(z - X'(Xz - y), λ)‖∞ ≤ 1e-12 max(1, ‖X'y‖∞)`,
     checked every 10 iterations and at the cap, at most `PB_REFERENCE_CAP` iterations. One pinned reference is
     generated per instance on Julia 1.13.4 (the harness refuses another version outside `PB_CHECK_ONLY`) and reused for
     both versions. Its file is named by the instance and the digests of `X` and `y` and records whether the run
     converged, the objective and the identities; `summarize.py` uses the objective only when the run converged and the
     digests, manifest and identities match.
   - `admm_untimed`: for each role and level, whether the target was reached within `PB_ADMM_CAP` iterations, the
     iteration count `k` (the first iteration that meets the target, or the number of iterations run, exactly the cap,
     when it does not), and the measures at `k`, with the trajectory at checkpoints. The levels are Boyd's rule and the
     stationarity `1e-8 max(1, ‖X'y‖∞)`. The stationarity check, with its two matrix-vector products, runs in this pass
     and in the reference pass at each check, and once after each timed run outside its timed interval.
   - `admm`: timed runs of those counts, `PB_ADMM_REPEATS` repeats per process in alternating role order, with the A/A
     copies. Each iteration of the timed loop evaluates Boyd's residual norms and tolerances (`O(p)`) and the number of
     iterations that meet the rule is consumed through a sink; the rule does not stop the loop. A run is valid when its
     last iteration meets Boyd's rule (level `boyd`) or its final stationarity meets the target (level `tight`), checked
     outside the timed interval. A role that did not reach a level is not timed there, and the comparisons that involve
     it are inconclusive (a lower bound); the other comparisons of the instance, step and level are kept. Reported
     per run: factorization time (construction and factorization), iteration time, total time, the objective
     `½‖Az - b‖² + λ‖z‖₁`, the final stationarity and the validity; the gap to the reference is computed by the summarizer.
6. **Prox outputs** (`outputs`): each primary role's output for each case, every step of `acc_gammas` and every input,
   from one untimed call per step and input, with the output of the A/A copy compared as bit patterns; written as binary
   files with SHA-256s for `accuracy.jl`. A failed refactor or prox is recorded for that step and input
   (`outputs-failures.tsv`), the object is rebuilt, and the other outputs of the case are still produced.

## Accuracy

`accuracy.jl` evaluates, from `X`, `y`, `v` and `γ` as stored and independently of the implementations:

- `η = ‖X'(Xβ - y) + ρ(β - v)‖∞ / (‖X'‖∞‖Xβ‖∞ + ‖X'y‖∞ + ρ(‖β‖∞ + ‖v‖∞))` in 256-bit BigFloat, with `ρ = 1/big(T(γ))`, and
  whether `η ≤ 1e3 eps(T)`, decided in BigFloat before `η` is rounded to Float64 for the file;
- the forward error for `n ≤ 500` (`PB_FORWARD_MAX_N`, recorded in the file header) against the 128-bit BigFloat Cholesky
  solution of the dual system `(XX' + ρI)a* = y - Xv`, `β* = v + X'a*`, relative to `‖β*‖∞` (absolute when `β* = 0`); a
  diagnostic that no criterion uses.

An output fails accuracy when `η > 1e3 eps(T)`, when it has a nonfinite entry (`η = Inf`) or when the harness recorded a
failure for it (`η = Inf` as well, with the failure as status). The table of `η` reports `η_C/η_S` beside pass and fail.
Accuracy failures are tabulated apart from timing. The accuracy evidence of a timing cell is the outputs at the steps and
inputs that the cell uses (first use: step 1 and the random inputs; warm: its step and kind of inputs; penalty: the
steps of its schedule and the random inputs; ADMM: the step `1/ρ` and the ADMM iterates); a cell whose variant or S
fails there has its timing reported and is excluded from the speed criteria, and a cell without that evidence is unknown.

The calibration of S before the campaign is the full outputs run and evaluation (step 5 of the rerun) on every main
case, before the timed processes; the summary then lists each main case where an S output exceeds `η/eps = 100` (a
factor 10 below the threshold) and counts the accuracy keys that have no evidence. The threshold is not changed after the
fact. `PB_CHECK_ONLY` runs the algebra check of the prototypes on one small case (`η ≤ 1e3 eps` for `γ ≤ 1`; the value at
`γ = 1e3` is reported) and a subset of the groups; it is not that calibration.

## Conditions

The campaign runs on exclusive nodes of one CPU kind, with the fixed design of two Julia versions, 1.10.13 and 1.13.4,
with `--threads=1 --check-bounds=no` and one BLAS thread: for each version, three fresh processes per role order
(`S_first`, `C_first`), 21 rounds each, and 5 ADMM repeats. The criteria need the evidence of both versions. An 8-thread
run is a diagnostic of first use and the warmed prox at 500×5000 and 1500×5000:
`PB_BLAS_THREADS=8 PB_GROUP=first_use,warm PB_FILTER='^f64/gauss/noisy/(500x5000|1500x5000)$'` (Julia still runs with one
thread); its files are summarized in a directory of their own, and the summarizer reports them apart from the criteria.
The working set of each object (`workingset`, from the groups that hold objects) and the load average at the start of
each case are recorded in the timing files, and the L3 size in the header, for comparison.

## Criteria

Pre-registered in the plan; the summarizer applies them as follows. Its design is fixed in `summarize.py` and in the
manifest: both Julia versions, the three processes of each order with one BLAS thread, 21 rounds (5 repeats), the A/A
copies, and every timed case, step, kind of input and ADMM instance of the manifest.

- Per cell (case, measurement, parameter) and variant `V ∈ {C_B, C_D, C_W}`, each of the six processes gives the per-round
  paired ratios `V/S`, `S'/S` and `V'/V` and their medians (rounds are paired within the process).
- **Faster**: all six per-process medians of `V/S` are below 0.95 and the distance of the largest median from 1 exceeds the
  A/A spread, the largest `|median - 1|` of `S'/S` and `V'/V` over the processes of the cell. **Slower** is the
  symmetric condition above 1.05. Anything else is **inconclusive**, not equal. A cell that lacks a process, a round or
  an A/A pair is **unknown**: its numbers are shown, and it is not labelled faster or slower.
- Evidence that is absent, incomplete, truncated, or made from another manifest or identity is **unknown**. An unknown
  criterion decides neither way, and the verdict is marked provisional while any evidence is missing.
- Recommendation per variant, from the Float64 cells with one BLAS thread; each criterion over cells is evaluated over
  the cells of both Julia versions:
  - **(A)** propose a PB fixed-penalty backend as a follow-up design issue: `V` passes `η` at each case, step and
    input of the manifest (Float32, rank-deficient, near-square and the #1078 controls included) where S passes; it is
    not slower on the warmed prox in any primary cell; its whole-ADMM total is faster on §11.1 (1500×5000) at `ρ = 1` at
    both stopping levels, with the objective gap at most that of S plus `1e-8 max(1, |p*|)`.
  - **(B)** solver-side factorization only, when `V` fails (A) but passes `η` at the §11.1 inputs at step 1, is faster in
    whole-ADMM total at both levels and is not slower at the tight target in any instance and `ρ`.
  - **(C)** no change otherwise.
  - The recommendation does not close #1078; the summarizer tabulates the control residuals of each variant against S.
- Exclusion for accuracy is reported apart from the timing label and does not count as a timing result in (A) and (B).
- Cases, tolerances and criteria do not change after the results are seen.

## Rerun

Run from the package root with `julia --threads=1 --check-bounds=no --project=.` (Julia 1.13.4 for the generator and the
reference). Replace `<inputs dir>`, `<output dir>`, `<reference dir>`, `<source dir>` and the two commits; set
`SHAS="PB_SOURCE_SHA=<sha of the source> PB_HARNESS_SHA=<sha of this folder>"` and `PB_SOURCE=<source dir>` in the
environment of every command below (the examples show them as `$SHAS`).

1. Smoke checks (seconds to minutes; the SHAs may be left out with `PB_CHECK_ONLY=1`; they exercise the prototypes against
   S, the integrity checks and a subset of the groups, not the campaign):

   ```
   PB_CHECK_ONLY=1 PB_INPUTS=<small inputs dir> julia ... docs/audit/wide-leastsquares-solves-1103/generate.jl
   PB_CHECK_ONLY=1 PB_INPUTS=<small inputs dir> PB_OUTPUT=<small output dir> julia ... docs/audit/wide-leastsquares-solves-1103/harness.jl
   PB_CHECK_ONLY=1 PB_INPUTS=<small inputs dir> PB_OUTPUTS=<small output dir>/outputs-<julia> julia ... docs/audit/wide-leastsquares-solves-1103/accuracy.jl
   ```

2. Inputs, once, on Julia 1.13.4: `$SHAS PB_INPUTS=<inputs dir> julia ... generate.jl` (`PB_PROFILE=small` for the small
   profile, `PB_FILTER=<regex>` for a subset of the cases). Every later command reads this directory.
3. Reference optima, once on Julia 1.13.4, and the iteration counts, once per Julia version (`PB_FILTER` can select an
   instance, for example `1500x5000`):

   ```
   $SHAS PB_GROUP=admm_reference PB_INPUTS=<inputs dir> PB_OUTPUT=<output dir> PB_REFERENCE=<reference dir> julia ... harness.jl
   $SHAS PB_GROUP=admm_untimed   PB_INPUTS=<inputs dir> PB_OUTPUT=<output dir> PB_REFERENCE=<reference dir> julia ... harness.jl
   ```

4. Timed processes, each a fresh process, for `PB_ORDER` in `S_first`, `C_first` and `PB_PROC` in 1, 2, 3 (the groups can be
   split over processes, for example `PB_GROUP=first_use,warm` or `PB_GROUP=admm`, with the same `PB_PROC`):

   ```
   $SHAS PB_ORDER=S_first PB_PROC=1 PB_ROUNDS=21 PB_GROUP=first_use,warm,penalty,alloc,admm \
   PB_INPUTS=<inputs dir> PB_OUTPUT=<output dir> PB_REFERENCE=<reference dir> julia ... harness.jl
   ```

5. Prox outputs and accuracy (the calibration of S), per Julia version; the controls alone with `PB_FILTER=c1078`:

   ```
   $SHAS PB_GROUP=outputs PB_INPUTS=<inputs dir> PB_OUTPUT=<output dir> julia ... harness.jl
   $SHAS PB_INPUTS=<inputs dir> PB_OUTPUTS=<output dir>/outputs-<julia> julia ... accuracy.jl
   ```

6. Summary: gather the output directories of all processes, with `inputs-index.tsv`, into `<summary dir>` (one summary
   directory per BLAS thread count; the manifest may also be passed as the second argument):

   ```
   python3 -I docs/audit/wide-leastsquares-solves-1103/summarize.py <summary dir> [<inputs dir>]
   ```

   It prints markdown tables and writes `tables.md`, `per-process-ratios.tsv` and `verdict.json` into `<summary dir>`.

Environment variables of `harness.jl`: `PB_INPUTS`, `PB_OUTPUT` (required unless `PB_CHECK_ONLY=1`), `PB_REFERENCE`
(directory of the reference optima and the iteration counts; default `PB_OUTPUT`), `PB_ORDER`, `PB_PROC`, `PB_ROUNDS`
(default 21), `PB_BATCHES` (3), `PB_GROUP` (comma-separated subset of `first_use`, `warm`, `penalty`, `alloc`, `outputs`,
`admm_reference`, `admm_untimed`, `admm`), `PB_FILTER` (regular expression on case ids), `PB_BLAS_THREADS` (1),
`PB_ADMM_REPEATS` (5), `PB_ADMM_CAP` (20000), `PB_REFERENCE_CAP` (100000), `PB_SOURCE`, `PB_SOURCE_SHA`,
`PB_HARNESS_SHA`, `PB_CHECK_ONLY`.

## Results

Not yet available: the campaign has not run. After it, the tables of `summarize.py` and the recommendation are added here.
