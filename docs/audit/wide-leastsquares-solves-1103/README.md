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
| `summarize.py` | checks that a campaign is the registered one (fail closed), rebuilds the tables from the raw rows and applies the criteria; `--tables-only` prints tables of partial data without criteria (Python standard library only) |
| `common.jl` | shared: the package modules, the prototypes, the roles, the manifest reader, the ADMM loop |

The scripts use Julia standard libraries and run with `--project=.` at the package root. Raw output goes to
directories outside the repository; no data or output is committed.

## Roles

All roles are loaded in one process. Prototype modules and the package source are included twice, into modules
with different names, so that the A/A copies are compiled separately.

All three prototypes use `ρ = inv(T(γ))` as `src/functions/leastsquares.jl` does, with `T` the element type of `X` and
`y`, and form the shift `I + XX'/ρ` with the scalar `inv(ρ)`, computed once per factorization as `lasso.m` forms `1/ρ`. They differ in what
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

`generate.jl` writes dense Float64 and Float32 main cases with `n < p`, and `X` and `y` of the same element type, and the
accuracy controls (small designs, among them a square and a tall one).

- Aspect ratios `n × p` of the main cases: 100×1000, 500×5000, 1000×2000, 1000×1100 (near-square), 200×20000, 1500×5000.
- Designs, each scaled to unit-norm columns in floating point: Gaussian; rank `n/2` (the rounded product of two Gaussian
  factors); duplicated rows (the first `n/2` rows repeated; the rows are equal by construction, so `XX'` is singular in
  exact arithmetic). The rank-deficient designs have a consistent response `y = Xβ*` and an inconsistent one
  `y = Xβ* + noise`; the Gaussian design has `y = Xβ* + noise`. For the product design the stored `X` and `y = Xβ*` come
  from rounded arithmetic, so their rank and consistency hold up to rounding; the validator of `generate.jl`
  (`PB_CHECK_ONLY=1`) tests the numerical rank and the distance of `y` from the range for the rank-deficient cases with
  `n ≤ 200`. `β*` has a nonzero `N(0, 1)` entry with probability `min(1, 100/p)` (about 100 nonzeros for `p ≥ 100`) and
  the noise has variance `1e-3`, as in the wide lasso of Boyd et al., §11.1.
- Steps: `γ ∈ {1e-3, 1, 1e3, 1e6}` for Float64 and `{1e-3, 1, 1e3}` for Float32. The lasso instances are also evaluated
  for accuracy at `γ = 0.1` and `10` (the ADMM penalties `ρ = 10` and `0.1`); the manifest column `acc_gammas` lists the
  evaluated steps and `gammas` the timed ones. The manifest stores the largest singular value `σ_max` of each stored `X`.
  The summary reports `ρ/σ_max²` for each main case and step with the penalty `ρ = inv(T(γ))` that the roles use (for
  Float32, the step and its reciprocal rounded to Float32).
- Proximal inputs: eight columns per main case. The first four are random `N(0, 1)` vectors and the last four are `z - u`
  iterates of an S-ADMM run on the same data at `ρ = 1` (`λ = 0.1‖X'y‖∞`) after 1, 5, 20 and 100 iterations. The study
  uses these four stored vectors as the ADMM inputs at every step, also for the accuracy evidence of the ADMM cells at
  `ρ = 0.1` and `10`, where they are registered test vectors rather than iterates of that run. The generator checks
  that the four vectors of each kind are pairwise different, and the harness checks it again when it reads a case.
- The wide lasso instances are the Float64 Gaussian cases 1500×5000 (§11.1), 500×5000 and 200×20000. The timing
  groups use the Gaussian cases and the consistent responses of the rank-deficient designs; their inconsistent
  responses are used for the accuracy evaluation.
- Accuracy controls: the reproducer of #1078 (`X = [1 0]`, `y = [1]`, `v = [±1/ρ, 0]`, `ρ = eps(T)/8`) and neighbours:
  `ρ ∈ {eps/8, eps, 32 eps}`, both signs, null-space components in `v`, 1×3 and 2×4 coordinate and diagonal designs
  (one with a zero singular direction), rotated 1×2 and 2×4 designs (full and deficient rank), a square and a tall
  design, in Float64 and Float32 (120 controls). They are deterministic and are used for the accuracy evaluation, not
  for timing.

## Integrity and provenance

- Every binary input is read from the bytes that were hashed: the file is read once, its SHA-256 is compared with the
  manifest, and the array is decoded from the same buffer. A mismatch or a size that differs from the manifest stops the
  process. `accuracy.jl` reads the prox outputs the same way, after its preflight has checked every output file.
- The manifest, the timing, untimed, outputs, failures and accuracy files end with the line `# complete: yes`, and the
  reference files with `complete yes`; the readers of this folder reject such a file without it, so an interrupted
  write is not mistaken for a result.
- Identities: the manifest header carries the source and harness labels and the Julia version of the generator (its own
  SHA-256 is computed by its readers). The timing, untimed, outputs and failures headers and the reference files carry
  `source_sha`, `harness_sha`, `inputs_index_sha256` and the Julia version; the timing headers also carry the role order,
  the process id, the conditions (rounds, batches, ADMM repeats and cap, Julia and BLAS threads, bounds checking), the
  BLAS configuration and OpenBLAS version, the CPU vendor and model, the L3 size and the load average, and the identity
  (file name and SHA-256) of each untimed file the ADMM stage used. The accuracy header carries the identities of the
  outputs and the SHA-256 of the outputs index it evaluated. Binary outputs have their SHA-256 in the outputs index.
- `PB_SOURCE_SHA` and `PB_HARNESS_SHA` are labels that the caller passes (7 to 40 lowercase hex digits, required outside
  `PB_CHECK_ONLY`). The scripts do not compare them with a checkout, also when one with `.git` is present; the campaign
  pins them, and the summarizer compares every file with the pinned values.
- The ADMM stage accepts the untimed counts of an instance only from the untimed file with the format, Julia version,
  source, harness, manifest digest, `X` and `y` digests, case (header and each row), cap and tight target of its own
  process, with every (ρ, primary role, level) present once, the status `reached` (`1 ≤ k ≤ cap`) or `capped` (`k = cap`)
  and no failed run; otherwise the case gets an `error` row.
- `accuracy.jl` runs a preflight before scoring: the outputs index and the failures index together (formats, columns,
  identities against each other, the manifest and `PB_SOURCE_SHA`/`PB_HARNESS_SHA`, one row per case and primary role
  with the element type, steps, inputs and shape of the manifest, `nfail` against the failure rows, failure keys in
  range and not repeated), then the size and SHA-256 of every output file. It lists every problem and writes no score
  when one is found.

## Measurements

A process with timing groups writes one file `timings-<julia>-<order>-<proc>-<groups>.tsv`: a header of `# key: value`
lines, one row per (case, measurement, parameter, round, position, role) and the closing line. The timed rows hold the
time per call, the GC time, the bytes and the allocations per call. Other rows hold context: `batch_plan` (the calls per
batch, batches and call cap, with the slowest calibrated time per call), `penalty_pairs` (the executed (step, input)
pairs of each schedule), `loadavg_1m`, `workingset` (the reachable bytes of each object; the same key can repeat across
groups) and, in the ADMM stage, `admm_iters` and `admm_reached`. A failure of one sample, of one role's setup or
calibration, or of a whole case is recorded as a row `error` (the cell and the message) and the process continues; the
summarizer refuses a campaign with an `error` row until that cell is rerun. A warm-up for each element type calls the
routines of each role (the ADMM routines for Float64 only) before the first timed sample; first use is therefore timed
in a warmed process, with the state of the native libraries shared.

**Pairing.** In the warm and penalty groups, within a (case, cell), all roles run batches with the same number of calls
along the same sequence of steps and inputs, and take the minimum over the same number of batches (`PB_BATCHES`, 3). A
batch is a set of complete cycles: a multiple of the number of inputs (4) and, for the penalty groups, of the length of
the step schedule, restarted at the first input and the first step; in the penalty group each object enters a batch at
the last step of the schedule. Scratch buffers and the state of the native libraries are not reset between batches or
roles. `PB_ORDER` (`S_first` or `C_first`) fixes the order of the roles in the odd rounds (or ADMM repeats) of a process
and the order is reversed in the even ones. The calls per batch come from a calibration of the primary roles: for warm,
once per case, on a complete cycle of the random inputs at `γ = 1`, shared by every step and both kinds of inputs of the
case; for penalty, once per schedule. The plan is the smallest multiple of the cycle with about 20 ms of calls and at
least 30 calls (warm) or about 50 ms of calls (penalty) for the slowest calibrated role, at most 3000 calls (rounded
down to a multiple of the cycle), and one cycle with one batch when that role needs 0.1 s or more per call. Each role is
calibrated under the same guard as the samples: a role whose calibration throws gets an `error` row for the cell and a
fresh object, the plan comes from the other roles, and the other roles, cells and schedules continue.

1. **First use** (`first_use`): construction, factorization and first prox on a fresh object at `γ = 1`, one sample per
   role and round, the input rotating through the four random vectors by round (the same for all roles of a round),
   `GC.gc()` before each sample outside the timed region, with GC time, bytes and allocations from `@timed`. The seconds of
   the setup steps of the primary roles are recorded too (`first_use_part`), on a separately sampled path: for S, a
   reconstruction of the cached SVD (the copy with the finiteness check, `svd`, the materialized `V`, and `s²` with `U'y`
   scaled by the singular values; the loss constructor and the allocation of its work vector are not included); for the
   C roles, the kernels of the constructor (the finiteness check and allocation, `XX'`, `X'y` for C_B, the
   factorization, and for C_W the copy of `X`, `trsm` and the triangular solve of `y`).
2. **Warmed prox** (`warm`): for each timing step and each kind of input (the four random vectors, the four ADMM
   iterates), the minimum over the batches of complete input cycles, per role and round.
3. **Penalty change** (`penalty`): S calls with the step of the schedule; each C role refactors from the cached `XX'` at
   that step and then calls the prox. No two consecutive calls (also across batches) use the same step. Two schedules,
   both of registered steps: alternating between `γ = 1` and `1e3` (`alt2`), and cycling through `1e-3, 1e3, 1, 1e6` for
   Float64 and `1e-3, 1e3, 1` for Float32 (`newstep`). Call `i` of a batch uses the step `mod1(i, G)` of the schedule and
   the input `mod1(i, 4)`, so one cycle executes the pairs (1, 1), (1e3, 2), (1, 3), (1e3, 4) for `alt2`, one input per
   step for the Float64 `newstep`, and every (step, input) pair for the Float32 `newstep`; the rows `penalty_pairs`
   record them.
4. **Warmed allocations** (`alloc`, timed cases only): `warmed_bytes(op::F) where {F} = (op(); op(); @allocated op())`
   behind a function barrier, for the prox cycle and for the refactor-and-prox cycle, per call, for each role, with the
   reachable bytes of each object.
5. **Complete ADMM runs** (`admm_reference`, `admm_untimed`, `admm`), Float64, on the three lasso instances. One loop
   serves all roles; only the x-update differs: `prox!(x, obj, z - u, 1/ρ)`. It follows Boyd's lasso loop with `α = 1`,
   `λ = 0.1‖X'y‖∞`, `z = soft(x + u, λ/ρ)`, `u += x - z` and the stopping rule with `ε_abs = 1e-4`, `ε_rel = 1e-2`.
   - `admm_reference`: S-ADMM at `ρ = 1` only, to the lasso stationarity
     `‖z - soft(z - X'(Xz - y), λ)‖∞ ≤ 1e-12 max(1, ‖X'y‖∞)`, checked every 10 iterations and at the cap, at most
     `PB_REFERENCE_CAP` iterations. One pinned reference is generated per instance on Julia 1.13.4 (the harness refuses
     another version outside `PB_CHECK_ONLY`) and serves both versions. Its file is named by the instance and the
     digests of `X` and `y` and records whether the run converged, the objective and the identities.
   - `admm_untimed`, at `ρ ∈ {0.1, 1, 10}`: for each primary role and level, the status (`reached`; `capped` when the run
     went through `PB_ADMM_CAP` iterations without meeting the target; `error` with the message when the run threw),
     the iteration count `k` (the first iteration that meets the target, or the cap) and the measures at `k`, with the
     trajectory at checkpoints. The levels are Boyd's rule and the stationarity `1e-8 max(1, ‖X'y‖∞)`. The A/A copies
     use the counts of their family. The stationarity check, with its two matrix-vector products, runs in this pass and
     in the reference pass at each check, and once after each timed run outside its timed interval.
   - `admm`, at the same `ρ`: timed runs of those counts, `PB_ADMM_REPEATS` repeats per process in alternating role
     order, with the A/A copies. Each iteration of the timed loop computes Boyd's residual norms (`O(p)`) and evaluates
     the rule (the dual comparison only when the primal one holds); the number of iterations that meet it is consumed
     through a sink, and the rule does not stop the loop. A run is valid when its last iteration meets Boyd's rule
     (level `boyd`) or its final stationarity meets the target (level `tight`), checked outside the timed interval. A
     capped family is not timed at that level; its comparisons are lower bounds. Reported per run: factorization time
     (construction and factorization), iteration time, total time, the objective `½‖Az - b‖² + λ‖z‖₁`, the final
     stationarity and the validity; the summarizer computes the gap to the reference of the same instance.
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

An output fails accuracy when `η > 1e3 eps(T)`. An output that the harness recorded as a failure has `η = Inf`, fails and
has the status `error: <message>`; an output with a nonfinite entry and no recorded failure has `η = Inf`, fails and has
the status `nonfinite`. For `n ≤ 500` the forward error of those two kinds is `Inf`; for larger `n` it is `NaN` for every
output. The per-case accuracy table reports the largest `η_C/η_S` over the (step, input) keys where `η_S > 0` and `η_C` is
finite, beside the failure counts; its row listing is capped at 40 cases.

The accuracy evidence of a timing cell is the (step, input) pairs that its measurement executes: first use, step 1 with
the four random inputs; warm, its step with its four inputs; penalty, the pairs of one cycle of the schedule (above); ADMM,
the step `1/ρ` with the four stored ADMM iterates. A cell whose variant or S fails at one of them has its timing reported
and is excluded from the speed criteria (it is not faster).

The calibration of S before the campaign is the full outputs run and evaluation (step 3 of the rerun) on every case,
before the timed processes; the summary lists each main case where an S output exceeds `η/eps = 100` (a factor 10 below
the threshold). The threshold is not changed after the fact. `PB_CHECK_ONLY` runs the algebra check of the prototypes on
one small case (`η ≤ 1e3 eps` for `γ ≤ 1`; the value at `γ = 1e3` is reported) and a subset of the groups; it is not that
calibration.

## Conditions

The campaign runs on exclusive nodes of one CPU kind, with the fixed design of two Julia versions, 1.10.13 and 1.13.4,
with `--threads=1 --check-bounds=no` and one BLAS thread: for each version, three fresh processes per role order
(`S_first`, `C_first`), 21 rounds each, and 5 ADMM repeats. An 8-thread run is a diagnostic of first use and the warmed
prox at 500×5000 and 1500×5000 (`PB_BLAS_THREADS=8 PB_GROUP=first_use,warm` with a filter on those cases; Julia still runs
with one thread); its files are kept apart from the campaign and summarized with `--tables-only`. The working set of each
object and the load average at the start of each case are recorded in the timing files, and the L3 size in the header,
for comparison.

## Criteria

Pre-registered in the plan. `summarize.py` applies them only to the registered campaign, in verdict mode:

- It takes the campaign directory and the pinned identity (`--source-sha`, `--harness-sha`, `--manifest`, the
  `inputs-index.tsv` whose SHA-256 it computes). It accepts the full inventory of cases and controls of the full profile
  (written into `summarize.py` from the plan and compared with the manifest), both Julia versions, the three processes of
  each order with one BLAS thread, 21 rounds and 5 ADMM repeats with all eight roles, one untimed file per version and
  instance, one converged reference per instance, and per version one outputs index, one failures index and one
  accuracy file of every case.
- Before computing anything it runs a preflight in three classes: the manifest and the pins; each file on its own
  (format, closing line, identity, conditions, parseable fields, values in their domain, no duplicate row, no `error`
  row); the inventory and the consistency across files (missing or extra files and rows, the role order of each round
  against the process order, the ADMM counts and status against the untimed file recorded in the timing header, a valid
  end point for every timed role and repeat, the accuracy rows against the outputs and failures). At the first class
  with a problem it lists every problem of that class, exits with status 2 and writes no verdict. A small-profile,
  filtered or partial campaign is refused; `--tables-only` prints the tables that its files allow, labelled partial,
  with no criteria and no recommendation, leaving out the files with a problem and listing the gaps.
- Per cell (case, measurement, parameter) and variant `V ∈ {C_B, C_D, C_W}`, each of the six processes of a version gives
  the per-round paired ratios `V/S`, `S'/S` and `V'/V` and their medians.
- **Faster**: all six per-process medians of `V/S` are below 0.95 and the distance of the largest median from 1 exceeds the
  A/A spread, the largest `|median - 1|` of `S'/S` and `V'/V` over the processes of the cell. **Slower** is the
  symmetric condition above 1.05. Anything else is **inconclusive**, not equal. An ADMM cell where S or `V` is capped is
  **capped**, a lower bound, which counts as inconclusive. A cell that fails accuracy (above) is **excluded**.
- Recommendation per variant, from the Float64 cells with one BLAS thread; each criterion over cells is evaluated over
  the cells of both Julia versions:
  - **(A)** propose a PB fixed-penalty backend as a follow-up design issue when all hold: A1, `V` passes `η` at each case,
    step and input of the manifest (Float32, rank-deficient, near-square and the #1078 controls included) where S
    passes; A2, it is not slower on the warmed prox in any Float64 cell; A3, its whole-ADMM total is faster on §11.1
    (1500×5000) at `ρ = 1` at both stopping levels, and its largest timed objective gap to the reference is at most that
    of S plus `1e-8 max(1, |p*|)` (false when S or `V` is capped there).
  - **(B)** solver-side factorization only, when (A) does not hold and all of these hold: B1, `V` passes `η` at the §11.1
    inputs at step 1; it is faster in whole-ADMM total there at both levels (the speed part of A3); B3, it is not slower
    at the tight target in any instance and `ρ`.
  - **(C)** no change when both (A) and (B) are ruled out: a criterion of (A) is false and a criterion of (B) is false.
  - The recommendation does not close #1078; the summarizer tabulates the control residuals of each variant against S.
- Cases, tolerances and criteria do not change after the results are seen.

## Rerun

Run from the package root. Set the pinned identity and the directories once in the shell, replacing the placeholders;
run each Julia block once per Julia version, with that version's `julia` first on the `PATH` (the generator and the
reference on 1.13.4 only):

```
export PB_SOURCE_SHA=<sha of the measured source> PB_HARNESS_SHA=<sha of this folder> PB_SOURCE=<checkout of the source>
export INPUTS=<inputs dir> OUT=<output dir of this Julia version> REF=<reference dir> D=docs/audit/wide-leastsquares-solves-1103
```

1. Smoke checks (seconds to minutes; they exercise the prototypes against S, the integrity checks and a subset of the
   groups, not the campaign; the SHAs may be unset with `PB_CHECK_ONLY=1`):

   ```
   PB_CHECK_ONLY=1 PB_INPUTS=$OUT/small-inputs julia --threads=1 --project=. $D/generate.jl
   PB_CHECK_ONLY=1 PB_INPUTS=$OUT/small-inputs PB_OUTPUT=$OUT/small julia --threads=1 --check-bounds=no --project=. $D/harness.jl
   PB_CHECK_ONLY=1 PB_INPUTS=$OUT/small-inputs PB_OUTPUTS=$OUT/small/outputs-$(julia -e 'print(VERSION)') julia --threads=1 --project=. $D/accuracy.jl
   ```

2. Inputs, once, on Julia 1.13.4 (the full profile; every later command reads this directory):

   ```
   PB_INPUTS=$INPUTS julia --threads=1 --project=. $D/generate.jl
   ```

3. Prox outputs and accuracy, per Julia version: the calibration of S, before the timed processes. The controls alone can
   be rerun with `PB_FILTER=c1078` into an output directory of their own, so that they do not replace the full index:

   ```
   PB_GROUP=outputs PB_INPUTS=$INPUTS PB_OUTPUT=$OUT julia --threads=1 --check-bounds=no --project=. $D/harness.jl
   PB_INPUTS=$INPUTS PB_OUTPUTS=$OUT/outputs-$(julia -e 'print(VERSION)') julia --threads=1 --project=. $D/accuracy.jl
   PB_GROUP=outputs PB_FILTER=c1078 PB_INPUTS=$INPUTS PB_OUTPUT=$OUT/controls julia --threads=1 --check-bounds=no --project=. $D/harness.jl
   ```

4. Reference optima, once on Julia 1.13.4, and the iteration counts, per Julia version:

   ```
   PB_GROUP=admm_reference PB_INPUTS=$INPUTS PB_OUTPUT=$OUT PB_REFERENCE=$REF julia --threads=1 --check-bounds=no --project=. $D/harness.jl
   PB_GROUP=admm_untimed PB_INPUTS=$INPUTS PB_OUTPUT=$OUT PB_REFERENCE=$REF julia --threads=1 --check-bounds=no --project=. $D/harness.jl
   ```

5. Timed processes, per Julia version, each a fresh process (the groups can be split over processes with the same
   `PB_ORDER` and `PB_PROC`, for example `PB_GROUP=first_use,warm,penalty,alloc` and `PB_GROUP=admm`, as long as each
   group runs once per process):

   ```
   for order in S_first C_first; do
     for proc in 1 2 3; do
       PB_ORDER=$order PB_PROC=$proc PB_INPUTS=$INPUTS PB_OUTPUT=$OUT/timings-$order-$proc PB_REFERENCE=$REF \
         julia --threads=1 --check-bounds=no --project=. $D/harness.jl
     done
   done
   ```

6. Summary: gather the output directories of both Julia versions and the reference directory into `<summary dir>`
   (without the 8-thread diagnostic and the controls-only outputs), then:

   ```
   python3 -I $D/summarize.py <summary dir> --source-sha $PB_SOURCE_SHA --harness-sha $PB_HARNESS_SHA --manifest $INPUTS/inputs-index.tsv
   python3 -I $D/summarize.py <diagnostic dir> --tables-only --manifest $INPUTS/inputs-index.tsv
   ```

   The first command writes `tables.md`, `per-process-ratios.tsv` and `verdict.json` into `<summary dir>` when the
   preflight passes, and otherwise lists the problems and exits with status 2. The second prints partial tables and
   writes `tables-partial.md`.

Environment variables of `harness.jl`: `PB_INPUTS`, `PB_OUTPUT` (required unless `PB_CHECK_ONLY=1`), `PB_REFERENCE`
(directory of the reference optima and the iteration counts; default `PB_OUTPUT`), `PB_ORDER`, `PB_PROC`, `PB_ROUNDS`
(default 21), `PB_BATCHES` (3), `PB_GROUP` (comma-separated subset of `first_use`, `warm`, `penalty`, `alloc`, `outputs`,
`admm_reference`, `admm_untimed`, `admm`), `PB_FILTER` (regular expression on case ids), `PB_BLAS_THREADS` (1),
`PB_ADMM_REPEATS` (5), `PB_ADMM_CAP` (20000), `PB_REFERENCE_CAP` (100000), `PB_SOURCE`, `PB_SOURCE_SHA`,
`PB_HARNESS_SHA`, `PB_CHECK_ONLY`.

## Results

Not yet available: the campaign has not run. After it, the tables of `summarize.py` and the recommendation are added here.
