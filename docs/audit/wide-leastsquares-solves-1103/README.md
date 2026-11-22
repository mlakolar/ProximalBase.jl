# Wide least-squares solves: cached thin SVD against cached Cholesky (#1103)

## Question

For dense wide least-squares problems (`n < p`), how does the cached thin SVD that `LeastSquaresLoss` uses for its
proximal operator compare with a cached Cholesky solve in observation space, at the same accuracy? The comparison covers
first use (construction, factorization and first prox), the warmed prox, a change of the step, whole ADMM lasso runs,
warmed allocations, and the accuracy of each prox from the original data, including the reproducer of
[#1078](https://github.com/mlakolar/ProximalBase.jl/issues/1078). It decides whether a production change is warranted;
it does not make one, and it does not change `src/` or `test/`.

This folder holds the tooling and, under Results, a summary of the campaign; the raw results are not committed. The
plan, with the reasoning behind each choice, is the comment
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
  `ρ = 0.1` and `10`, where they are registered test vectors rather than iterates of that run. The validator of the
  generator (`PB_CHECK_ONLY=1`) checks that the four vectors of each kind are pairwise different, and the harness checks
  it each time it reads a main case.
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
  pins them, and the summarizer compares the identity of each file it reads with the pinned values (the failures index
  through the outputs index it belongs to).
- The ADMM stage accepts the untimed counts of an instance only from the untimed file with the format, Julia version,
  source, harness, manifest digest, `X` and `y` digests, case (header and each row), cap and tight target of its own
  process, with every (ρ, primary role, level) present once, the status `reached` (`1 ≤ k ≤ cap`) or `capped` (`k = cap`)
  and no failed run; otherwise the case gets an `error` row.
- `accuracy.jl` runs a preflight before scoring: the outputs index and the failures index together (formats, columns,
  identities against each other, the manifest and `PB_SOURCE_SHA`/`PB_HARNESS_SHA`, one row per case and primary role
  with the element type, steps, inputs and shape of the manifest, `nfail` against the failure rows, failure keys in
  range and not repeated), then the size and SHA-256 of each output file. It stops after the header checks when one of
  them fails, and after the row and file checks otherwise, lists the problems it found and writes no score.

## Measurements

A process with timing groups writes one file `timings-<julia>-<order>-<proc>-<groups>.tsv`: a header of `# key: value`
lines, one row per (case, measurement, parameter, round, position, role) and the closing line. The timed rows hold the
time per call, the GC time, the bytes and the allocations per call. Other rows hold context: `batch_plan` (the calls per
batch, batches and call cap, with the slowest calibrated time per call), `penalty_pairs` (the executed (step, input)
pairs of each schedule), `loadavg_1m` and `workingset` (the reachable bytes of each object), each keyed by the group,
and, in the ADMM stage, `admm_iters` and `admm_reached`. A computation error of one sample, of one role's setup or
calibration, or of a whole case (including a refused untimed file) is recorded as a row `error` (the cell and the
message) and the process continues; the summarizer refuses a campaign with an `error` row until that cell is rerun. An
integrity error when a case's inputs are read (a SHA-256 or size mismatch) stops the process. A warm-up for each element type calls the
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
down to a multiple of the cycle), and one cycle with one batch when that role needs 0.1 s or more per call; the
`batch_plan` row records the slowest calibrated time per call losslessly, so that the summarizer recomputes the plan.
Each participating primary role (one with an object) is calibrated under the same guard as the samples: a role whose
calibration throws gets an `error` row for the cell and a rebuilt object (absent again if the rebuild fails too), the
plan comes from the other roles, and the other roles, cells and schedules continue.

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
     another version outside `PB_CHECK_ONLY`) and serves both versions. Its file name holds the instance and the first 12
     hexadecimal digits of the digests of `X` and `y`; the file records both full digests, whether the run converged,
     the objective, `λ`, `λmax`, the scale and the identities, and names the binary file of `z` (`p` Float64 values).
   - `admm_untimed`, at `ρ ∈ {0.1, 1, 10}`: for each primary role and level, the status (`reached`; `capped` when the run
     went through `PB_ADMM_CAP` iterations without meeting the target; `error` with the message when the run threw),
     the iteration count `k` (the first iteration that meets the target, or the cap) and the measures at `k` (Boyd's `r`,
     `s`, `eps_pri` and `eps_dual`, the stationarity and the objective), with the
     trajectory at checkpoints, at the reached iterations and at the cap. The levels are Boyd's rule and the stationarity `1e-8 max(1, ‖X'y‖∞)`. The A/A copies
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
- the forward error for `n ≤ PB_FORWARD_MAX_N` (default 500, recorded in the file header) against the 128-bit BigFloat Cholesky
  solution of the dual system `(XX' + ρI)a* = y - Xv`, `β* = v + X'a*`, relative to `‖β*‖∞` (absolute when `β* = 0`); a
  diagnostic that no criterion uses.

An output fails accuracy when `η > 1e3 eps(T)`. An output that the harness recorded as a failure has `η = Inf`, fails and
has the status `error: <message>`; an output with a nonfinite entry and no recorded failure has `η = Inf`, fails and has
the status `nonfinite`. For `n ≤ PB_FORWARD_MAX_N` the forward error of those two kinds is `Inf`; for larger `n` it is
`NaN` for every output. The accuracy header records the Julia version that produced the outputs (`julia`) and the one
that evaluated them (`evaluator_julia`). The per-case accuracy table reports the largest `η_C/η_S` over the (step, input) keys where `η_S > 0` and `η_C` is
finite, beside the failure counts; its row listing is capped at 40 cases.

The accuracy evidence of a timing cell is a set of (step, input) pairs: first use, step 1 with the four random inputs;
warm, its step with its four inputs; penalty, the pairs that one cycle of the schedule executes (above); ADMM, the step
`1/ρ` with the four stored ADMM vectors, registered test vectors sampled from an S-ADMM run at `ρ = 1` that the timed
runs do not take as inputs. A cell whose variant or S fails at one of them has its timing reported
and is excluded from the speed criteria (it is not faster).

The calibration of S before the campaign is the full outputs run and evaluation (step 3 of the rerun) on every case,
before the timed processes; `summarize.py --tables-only` on that outputs directory lists each main case where an S
output exceeds `η/eps = 100` (a factor 10 below the threshold), before any timed process runs. The threshold is not changed after the fact. With `PB_CHECK_ONLY`, the algebra check of
the prototypes uses one small case per element type (`η ≤ 1e3 eps` for `γ ≤ 1`; the value at `γ = 1e3` is reported),
and the default smoke run executes all eight harness groups on reduced selections (step 1 of the rerun); neither is that
calibration.

## Conditions

The plan runs the campaign on exclusive nodes of one CPU kind (Results states the nodes of the campaign that ran), with the fixed design of two Julia versions, 1.10.13 and 1.13.4,
with `--threads=1 --check-bounds=no` and one BLAS thread: for each version, three fresh processes per role order
(`S_first`, `C_first`), 21 rounds each, and 5 ADMM repeats. An 8-thread run is a diagnostic of first use and the warmed
prox at 500×5000 and 1500×5000 (`PB_BLAS_THREADS=8 PB_GROUP=first_use,warm` with a filter on those cases; Julia still runs
with one thread); its files are kept apart from the campaign and summarized with `--tables-only`. The working set of each
object and the load average at the start of each case are recorded in the timing files, and the L3 size in the header,
for comparison.

## Criteria

Pre-registered in the plan. `summarize.py` applies them in verdict mode, to a campaign that passes its checks.

**Scope of the checks.** The summarizer guards against incomplete, mixed, misplaced, truncated or corrupted evidence:
missing or extra files and rows, files of another identity, manifest, Julia version, CPU kind or condition, values out of
their domain, and values that contradict another value of the same record or of a related file (for example a pass flag
against its `η`, an ADMM status against its measures, a batch plan against its rule). The authenticity of the values
rests on the pinned harness and on the checksum list written when the campaign is gathered; a hand-made edit that keeps
the files consistent with each other is outside these checks.


- It takes the campaign directory and the pinned identity (`--source-sha`, `--harness-sha`, `--manifest`, the
  `inputs-index.tsv` whose SHA-256 it computes). It accepts the full inventory of cases and controls of the full profile
  (written into `summarize.py` from the plan and compared with the manifest), both Julia versions, the three processes of
  each order with one BLAS thread, 21 rounds with all eight roles, 5 ADMM repeats with the roles of the families that
  reached each level (a capped family has its level records and no timed rows), one untimed file per version and
  instance, one converged reference per instance, per version one outputs index, one failures index and one accuracy
  file of every case, one CPU vendor and model across the primary timing files and one BLAS configuration per Julia
  version, and the checksum list `sha256-list.tsv` of the gathered files, which must hold exactly the files it reads
  with the SHA-256 of the bytes it parses.
- Before computing anything it runs a preflight in three classes: the manifest, the pins and the checksum list; each
  file on its own and the files against the checksum list
  (format, closing line, identity, conditions, parseable fields, values in their domain, no duplicate row, no `error`
  row); the inventory and the consistency across files (missing or extra files and rows, the role order of each round
  against the process order, the ADMM counts and status against the untimed file recorded in the timing header, a valid
  end point for each timed role and repeat, the accuracy rows against the outputs and failures). At the first class
  with a problem it lists the problems it found in that class (a file with another format or other columns is not
  checked further), exits with status 2 and writes no verdict. A small-profile,
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

Run from the package root. Each selected Julia version needs the package root's project environment instantiated for it
(`--project=.` selects that environment; it does not install its dependencies). The commands of a block that depend on
each other are chained with `&&`, and step 5 runs in a subshell that exits with the status of a failed process, so that
a block stops at its first nonzero exit. A zero exit status does not mean that every sample succeeded: the harness
records a failed sample, setup, calibration or case as an `error` row and continues, the untimed file records a failed
run, and a reference can end capped and unconverged; verdict mode refuses a campaign with any of these. `OUT` is a new,
empty directory per campaign that holds the campaign output of all Julia versions and nothing else (no links); `SMOKE`
(the smoke checks) and `SUM` (the gathered campaign) lie outside `OUT`. The harness and `accuracy.jl` refuse to write a
file that exists already, so for launches that run one after the other each file in `OUT` comes from one attempt.

Set the pinned identity and the directories once in the shell, replacing the placeholders:

```
export PB_SOURCE_SHA="<commit of the measured package source>" PB_HARNESS_SHA="<commit of this folder pinned for the campaign>" PB_SOURCE="<checkout of that source commit>"
export INPUTS="<inputs dir>" OUT="<new empty campaign output dir>" SMOKE="<new smoke dir outside OUT>" SUM="<new summary dir outside OUT>"
export D=docs/audit/wide-leastsquares-solves-1103
```

Run each Julia block once per Julia version (the full-profile generator of step 2 and the campaign reference on 1.13.4 only), with that
version's `julia` first on the `PATH`. Each time the selected `julia` changes, set `V` from it again and check that the
command succeeded; the blocks below name their output directories by `$V`:

```
V="$(julia --startup-file=no -e 'print(VERSION)')" && [ -n "$V" ] && export V && echo "selected Julia $V"
```

1. Smoke checks, per Julia version (seconds to minutes; the SHAs may be unset with `PB_CHECK_ONLY=1`). They generate
   the reduced input profile `small` into a directory of this version, check the prototypes against S, and run all eight
   harness groups with 2 rounds and 2 ADMM repeats on the smallest selected case of each element type (for the ADMM
   groups the smallest Float64 lasso instance; the outputs group adds two controls), then the evaluator; they are not
   the campaign. A repeat uses a new `SMOKE` (the harness refuses to rewrite its files):

   ```
   PB_CHECK_ONLY=1 PB_INPUTS="$SMOKE/$V/inputs" julia --threads=1 --project=. "$D/generate.jl" &&
   PB_CHECK_ONLY=1 PB_INPUTS="$SMOKE/$V/inputs" PB_OUTPUT="$SMOKE/$V/out" julia --threads=1 --check-bounds=no --project=. "$D/harness.jl" &&
   PB_CHECK_ONLY=1 PB_INPUTS="$SMOKE/$V/inputs" PB_OUTPUTS="$SMOKE/$V/out/outputs-$V" julia --threads=1 --project=. "$D/accuracy.jl"
   ```

2. Inputs, once, on Julia 1.13.4 (the full profile). The campaign's Julia commands read `$INPUTS`; the gather copies its
   manifest into `$SUM`, and the verdict command reads that copy:

   ```
   PB_INPUTS="$INPUTS" julia --threads=1 --project=. "$D/generate.jl"
   ```

3. Prox outputs and accuracy, per Julia version: the calibration of S, before the timed processes. The third command
   (`--tables-only` on the outputs directory) prints, among its tables, the main cases where an S output exceeds
   `η/eps = 100`; inspect that table before step 5. The controls alone can be rerun with `PB_FILTER=c1078` into a
   directory outside `OUT`:

   ```
   PB_GROUP=outputs PB_INPUTS="$INPUTS" PB_OUTPUT="$OUT/$V" julia --threads=1 --check-bounds=no --project=. "$D/harness.jl" &&
   PB_INPUTS="$INPUTS" PB_OUTPUTS="$OUT/$V/outputs-$V" julia --threads=1 --project=. "$D/accuracy.jl" &&
   python3 -I "$D/summarize.py" "$OUT/$V/outputs-$V" --tables-only --manifest "$INPUTS/inputs-index.tsv"
   ```

4. The campaign reference, once on Julia 1.13.4 (outside `PB_CHECK_ONLY` the harness refuses this group on another
   version), and the iteration counts, per Julia version. The blocks are independent, both write to `$OUT/reference`,
   and each must exit with status 0:

   ```
   PB_GROUP=admm_reference PB_INPUTS="$INPUTS" PB_OUTPUT="$OUT/$V" PB_REFERENCE="$OUT/reference" julia --threads=1 --check-bounds=no --project=. "$D/harness.jl"
   ```

   ```
   PB_GROUP=admm_untimed PB_INPUTS="$INPUTS" PB_OUTPUT="$OUT/$V" PB_REFERENCE="$OUT/reference" julia --threads=1 --check-bounds=no --project=. "$D/harness.jl"
   ```

5. Timed processes, per Julia version, each a fresh process. The block exits with the status of the first failed
   process and starts no later one. The groups can be split over processes with the same `PB_ORDER` and `PB_PROC`
   (for example a block with `PB_GROUP=first_use,warm,penalty,alloc` and a block with `PB_GROUP=admm`), as long as
   each group runs once per process; each such block writes one timing file per slot:

   ```
   (
     for order in S_first C_first; do
       for proc in 1 2 3; do
         PB_ORDER="$order" PB_PROC="$proc" PB_INPUTS="$INPUTS" PB_OUTPUT="$OUT/$V/timings-$order-$proc" PB_REFERENCE="$OUT/reference" \
           julia --threads=1 --check-bounds=no --project=. "$D/harness.jl" || exit $?
       done
     done
   )
   ```

   **Retries.** A retry first removes the files of the failed attempt; the harness and `accuracy.jl` refuse to start
   while a file they would write exists. Per writer:
   - step 3, a failed evaluation: remove that version's `accuracy-*.tsv` from `PB_OUTPUTS`, keep the indices and
     binaries, rerun `accuracy.jl`, then run the third command and inspect its S-margin table before step 5; a failed
     third command alone: rerun just it; a failed outputs run: remove that version's `outputs-<julia>` directory and
     rerun the block;
   - step 4, a failed reference or untimed run: remove from `PB_REFERENCE` only the files that run writes (the
     `reference-*.tsv` files and their `-z.f64` binaries; that version's `admm-untimed-<julia>-*.tsv`), keeping the
     files of the other group and version, and rerun that block;
   - step 5, a failed process: remove from its `PB_OUTPUT` only the timing file of the failed (version, order, process,
     groups), keeping a sibling file of the same slot from another split-group block, rerun that slot with the same
     `PB_GROUP`, then the remaining slots of that block.

6. Run only after every applicable block has exited with status 0 (steps 3 and 5 and the untimed block of step 4 for
   both Julia versions, and the campaign reference block on 1.13.4), retried blocks and slots included, and after the
   cleanup that each retry needs. Gather the campaign into a new directory `$SUM` outside `OUT` and summarize. Duplicate,
   incomplete or inconsistent gathered evidence is refused by the gather or by verdict mode; files that the gather does
   not copy (for example the raw prox binaries) are not checked by it. `--gather` copies, from the producer tree `$OUT`,
   the files that verdict mode reads (by file-name pattern: the timing files into `timings/`, the untimed and reference
   files with the `z` binaries into `reference/`, and each `outputs-<julia>` directory's outputs index, failures index
   and accuracy file into a directory of the same name) and the manifest given by `--inputs`. It refuses a destination
   that exists and is not empty or lies inside the producer tree, a symbolic link below the producer root, another
   manifest in the tree, two files with the same destination and a kind of file the tree lacks, before copying anything;
   it stops at a failed copy; when all copies have succeeded it writes `sha256-list.tsv` (format `pb1103-checksums-1`:
   the path of each gathered file relative to `$SUM` and the SHA-256 of the copied bytes), to be kept with the raw
   results:

   ```
   python3 -I "$D/summarize.py" --gather "$OUT" "$SUM" --inputs "$INPUTS/inputs-index.tsv" &&
   python3 -I "$D/summarize.py" "$SUM" --source-sha "$PB_SOURCE_SHA" --harness-sha "$PB_HARNESS_SHA" --manifest "$SUM/inputs-index.tsv"
   ```

   The summary command, which runs only when the gather succeeded, writes `tables.md`, `per-process-ratios.tsv` and
   `verdict.json` into `$SUM` when the preflight passes, and otherwise lists the problems and exits with status 2. The
   8-thread diagnostic and other partial data are summarized apart with
   `python3 -I "$D/summarize.py" "<diagnostic dir>" --tables-only --manifest "$INPUTS/inputs-index.tsv"`, which writes
   `tables-partial.md`.

Environment variables of `harness.jl`: `PB_INPUTS`, `PB_OUTPUT` (required unless `PB_CHECK_ONLY=1`), `PB_REFERENCE`
(directory of the reference optima and the iteration counts; default `PB_OUTPUT`), `PB_ORDER`, `PB_PROC`, `PB_ROUNDS`
(default 21), `PB_BATCHES` (3), `PB_GROUP` (comma-separated subset of `first_use`, `warm`, `penalty`, `alloc`, `outputs`,
`admm_reference`, `admm_untimed`, `admm`), `PB_FILTER` (regular expression on case ids), `PB_BLAS_THREADS` (1),
`PB_ADMM_REPEATS` (5), `PB_ADMM_CAP` (20000), `PB_REFERENCE_CAP` (100000), `PB_SOURCE`, `PB_SOURCE_SHA`,
`PB_HARNESS_SHA`, `PB_CHECK_ONLY`.

## Results

The tables and numbers below are copied from the output of verdict mode of `summarize.py` at this commit, run on the
gathered campaign with the pins of the campaign (`--harness-sha` names the harness that produced the data):
`tables.md`, `verdict.json` and `per-process-ratios.tsv`, which are not committed. Each subsection names the `tables.md`
sections it comes from; several of them list only their first rows, and the text says when a number comes from such an
excerpt. The complete matrix of the 108 ADMM cells is in the untruncated section *Timing detail* and, per process, in
`per-process-ratios.tsv`. The time limits and reruns of the timing processes and the second verdict run come from the
record of the run, not from these files.

Ratios are `V/S`, the recorded time of the variant over that of S; a ratio below 1 is a lower recorded time. *Faster*,
*slower*, *inconclusive*, *capped* and *excluded* are the verdict labels of Criteria, with their stronger conditions:
for example the ratio 0.7877 of C_B (1.10.13, 1500×5000, `ρ = 0.1`, tight target) is inconclusive.

### Campaign and conditions

- Pins: source `fdf9b52777b280750a4402776ffeb94b7c702be6`, harness `72c8848f60c87d8f4101e46f6a0b5c1ca797c982`, manifest
  SHA-256 `7f2b2fb32581899672947c46d516f87a80cc298205d919e85d79430270fdd662`.
- Julia 1.13.4 and 1.10.13, `--threads=1 --check-bounds=no`, one BLAS thread; three processes per role order and
  version, 21 rounds and 5 ADMM repeats each. All twelve timing processes and the 8-thread diagnostic ran on one CPU
  kind, AMD EPYC 7513 (Zen 3), on a Slurm cluster.
- The plan has exclusive nodes (Conditions). By the maintainer's decision this campaign ran on nodes shared with other
  jobs. The load average at process start (1, 5 and 15 minutes) lies between 14.48 and 60.51 (section *Processes*).
  This is the main caveat on the timing tables. The A/A controls measure it: over the versions, measurements and
  pairs, the median `|S'/S - 1|` or `|V'/V - 1|` of the per-process medians is 0.001502 to 0.02015, and the largest
  single deviation is 0.6312 (1.10.13, warm, `C_W'/C_W`) (section *A/A controls*). These controls measure the
  variation between the two copies of a role, including what shared load adds to it; they do not isolate the effect of
  the other jobs. A cell is labelled faster when its six process medians are below 0.95 and the largest of them lies
  farther below 1 than the A/A spread of that cell, and slower when the six are above 1.05 and the smallest lies
  farther above 1 than that spread; a noisier cell therefore tends to come out inconclusive, but a load change that
  affects S and the variant differently within the paired rounds is not removed by this rule.
- Six of the twelve timing processes reached a 5 h limit in the 200×20000 ADMM runs and were rerun once from scratch
  with a 10 h limit, the partial file removed by the retry rule of step 5: 1.13.4 `C_first`-3, and 1.10.13 `S_first`-1,
  `S_first`-2, `S_first`-3, `C_first`-2 and `C_first`-3. The reruns ran at other times and on other nodes than their
  sibling processes.
- Verdict mode accepted the gathered campaign (exit status 0); a second run of verdict mode on the same directory gave
  an identical `verdict.json`.

### Recommendation

**(C), no change**, for C_B, C_D and C_W (section *Criteria and recommendation*). `#1078` stays open.

| variant | A1 accuracy where S passes | A2 warmed prox not slower (288 Float64 cells) | A3 speed, §11.1 at ρ = 1 | A3 objective gap | B1 accuracy at §11.1, step 1 | B3 not slower at the tight target (18 cells) |
|---|---|---|---|---|---|---|
| C_B | false: 1202 keys | false: 129 slower, 124 excluded, 35 inconclusive | true: 4 of 4 faster | true | true | false: 2 slower, 2 excluded, 2 capped, 4 inconclusive, 8 faster |
| C_D | false: 272 keys | false: 213 slower, 75 inconclusive | true: 4 of 4 faster | true | true | false: 4 slower, 2 capped, 4 inconclusive, 8 faster |
| C_W | false: 290 keys | false: 35 slower, 253 inconclusive | true: 4 of 4 faster | true | true | false: 4 slower, 2 capped, 2 inconclusive, 10 faster |

A key of A1 is a (Julia version, case, step, input) where the variant fails `η` and S passes. (A) is ruled out because
each variant fails A1 and A2. On the §11.1 instance (1500×5000) at `ρ = 1` each variant is faster for the whole ADMM run
at both levels, its largest objective gap is within that of S plus the registered slack (the printed gaps of S and the
variant agree: 0.00271 at Boyd's rule, 2.93e-09 at the tight target, both versions), and B1 holds; (B) is ruled out
because B3 fails: at the tight target on 200×20000 the variants are slower at `ρ = 1` and, for C_D and C_W, at `ρ = 0.1`
(for C_B that cell is excluded by accuracy).

### Accuracy

Sections *Accuracy: outputs, failures ... per role*, *Margin of S under the threshold* and *Accuracy domain per design*.
A failure is `η > 1e3 eps(T)`, a nonfinite output or a recorded failure. Where the two versions differ, the cell gives
1.13.4 / 1.10.13; otherwise one value holds for both.

| kind | role | outputs | failures | max η/eps | median η/eps |
|---|---|---|---|---|---|
| main | S | 1728 | 0 | 44.66 / 49.34 | 2.205 / 2.264 |
| main | C_B | 1728 | 573 | 3.44e+08 | 7.119 |
| main | C_D | 1728 | 97 | 9.88e+05 | 0.7431 |
| main | C_W | 1728 | 96 | 7.00e+05 | 0.7674 |
| control | S | 120 | 52 | 4.50e+15 | 12.57 |
| control | C_B | 120 | 68 | inf | 6.06e+05 |
| control | C_D | 120 | 91 | inf | 8.39e+06 |
| control | C_W | 120 | 101 | inf | 8.39e+06 |

S has no main case with an output above `η/eps = 100`; its largest main value is 44.66 on 1.13.4 and 49.34 on 1.10.13,
about a factor 20 below the threshold of 1000.

Where the variants fail, over the (case, step) pairs with accuracy evidence:

- C_B fails on each design in both element types: Float64 Gaussian 10 of 30 pairs, rank `n/2` 22 of 48, duplicated
  rows 22 of 48; Float32 3 of 18, 8 of 36 and 8 of 36. Its failing pairs reach `ρ/σ_max² = 8.27e-04` (Float64
  Gaussian).
- C_D and C_W have no failing pair on the Gaussian design or in Float32. On the Float64 rank-deficient designs C_D fails
  6 of 48 pairs (duplicated rows, up to `ρ/σ_max² = 1.65e-07`) and 7 of 48 (rank `n/2`, up to 1.81e-06), C_W 6 of 48
  (duplicated rows, up to 1.65e-07) and 6 of 48 (rank `n/2`, up to 1.09e-07); the smallest evaluated ratio there is
  4.38e-09 (duplicated rows) and 1.81e-09 (rank `n/2`). In the rows that section *Accuracy per main case* lists, their
  failures fall on the inconsistent responses (8 outputs per such case).

### Speed

Sections *Timing labels per measurement and variant*, *Timing detail* and *ADMM: total time ratio*. The first table
gives the median `V/S` over the cells of each (version, measurement, variant, element type) that have a ratio, as
1.13.4 / 1.10.13; it includes the cells excluded by accuracy and leaves out the capped ADMM cell, which has no ratio
(17 of the 18 ADMM cells of each row).

| measurement | type | C_B | C_D | C_W |
|---|---|---|---|---|
| first use | Float64 | 0.08546 / 0.08889 | 0.08411 / 0.08759 | 0.1639 / 0.1747 |
| first use | Float32 | 0.07306 / 0.07381 | 0.07206 / 0.07352 | 0.1288 / 0.1316 |
| warmed prox | Float64 | 1.33 / 1.622 | 1.325 / 1.616 | 0.9969 / 1.066 |
| warmed prox | Float32 | 1.394 / 1.566 | 1.396 / 1.536 | 0.9686 / 0.9825 |
| penalty change | Float64 | 5.879 / 5.628 | 5.899 / 5.651 | 61.44 / 60.19 |
| penalty change | Float32 | 7.404 / 6.965 | 7.325 / 7.012 | 69.74 / 68.63 |
| ADMM total | Float64 | 0.3906 / 0.5192 | 0.3868 / 0.4598 | 0.4212 / 0.4515 |

Each first-use cell is faster (18 of 18 per version, variant and type). In the warmed prox, C_B and C_D have no faster
cell, and C_W has 10 faster cells (1.13.4, Float32) and 71 slower ones of its 504. On the §11.1 instance at
`γ = 1` with random inputs, first use gives 0.08985 / 0.08806 (C_B), 0.08913 / 0.08763 (C_D) and 0.1732 / 0.1716 (C_W),
and the warmed prox 1.3 / 1.242, 1.292 / 1.269 and 0.9521 / 0.9859. In the penalty change each C role refactors at
each call and S does not.

Whole ADMM runs, total time `V/S` (median of the per-process medians, 1.13.4 / 1.10.13). Unmarked ratios are faster;
`i` inconclusive, `s` slower, `x` excluded by accuracy:

| instance | ρ | level | C_B | C_D | C_W |
|---|---|---|---|---|---|
| 500×5000 | 0.1 | Boyd | 0.3325 / 0.4702 | 0.3295 / 0.4598 | 0.3707 / 0.4206 |
| 500×5000 | 0.1 | tight | 0.8227 i / 1.097 i | 0.8267 i / 1.186 i | 0.7525 / 0.8963 i |
| 500×5000 | 1 | Boyd | 0.1424 / 0.1737 | 0.1414 / 0.1703 | 0.2208 / 0.2364 |
| 500×5000 | 1 | tight | 0.3906 / 0.5444 | 0.3868 / 0.4979 | 0.4212 / 0.4515 |
| 500×5000 | 10 | Boyd | 0.3001 / 0.5192 | 0.2971 / 0.4071 | 0.3479 / 0.4078 |
| 500×5000 | 10 | tight | 0.7459 / 1.182 i | 0.7377 i / 1.211 i | 0.7027 / 0.8236 i |
| 200×20000 | 0.1 | Boyd | 1.368 x / 1.291 x | 1.359 i / 1.24 i | 1.391 s / 1.244 s |
| 200×20000 | 0.1 | tight | 1.998 x / 1.938 x | 1.974 s / 2.035 s | 1.9 s / 1.949 s |
| 200×20000 | 1 | Boyd | 0.6258 / 0.5892 | 0.5983 / 0.5801 | 0.6492 / 0.6096 i |
| 200×20000 | 1 | tight | 1.762 s / 1.828 s | 1.741 s / 1.804 s | 1.724 s / 1.717 s |
| 200×20000 | 10 | Boyd | 0.4682 / 0.469 | 0.4757 / 0.4576 | 0.5288 / 0.4782 |
| 200×20000 | 10 | tight | capped | capped | capped |
| 1500×5000 | 0.1 | Boyd | 0.2039 / 0.2465 | 0.1987 / 0.2412 | 0.2432 / 0.2782 |
| 1500×5000 | 0.1 | tight | 0.676 / 0.7877 i | 0.6738 / 0.7703 | 0.5565 / 0.6289 |
| 1500×5000 | 1 | Boyd | 0.1064 / 0.1104 | 0.1064 / 0.1071 | 0.1837 / 0.1847 |
| 1500×5000 | 1 | tight | 0.2018 / 0.2588 | 0.197 / 0.2391 | 0.2447 / 0.2904 |
| 1500×5000 | 10 | Boyd | 0.1582 / 0.173 | 0.1575 / 0.1655 | 0.2158 / 0.2253 |
| 1500×5000 | 10 | tight | 0.3351 / 0.3642 | 0.3352 / 0.3789 | 0.3256 / 0.3679 |

At 200×20000, `ρ = 10`, tight target, S and each variant reached the cap of 20000 iterations in both versions. A
capped run gives a lower bound on its own time to the target; the cell is not timed, has no `V/S`, and the two bounds
give no bound on the ratio. The variants trade a cheaper factorization for usually dearer iterations: in the rows that
section *ADMM: total time ratio* lists (1.10.13), the factorization ratio is 0.04646 to 0.1776 and the iteration ratio
0.989 to 2.07. The slower totals are on the runs with many iterations: in the listed stopping records
(1.10.13, section *ADMM stopping records*) the tight target takes 17503 iterations at `ρ = 0.1` and 2361 at `ρ = 1` on
200×20000, against 103 at `ρ = 1` on 1500×5000, and the four roles stop at the same iteration in those records.

The 8-thread diagnostic (one `S_first` process per version, eight BLAS threads, Julia on one thread, first use and the
warmed prox on 500×5000 and 1500×5000; files `tables-partial.md`, section *Processes with BLAS threads other than 1*) is
not part of the verdict; A/A statistics and verdict labels are not reported for it. In the rows it lists (the Float32
duplicated-row cases, 40 of 288 per version) first use gives `V/S` 0.03017 to 0.1526, and the warmed prox 2.056 to 3.189
for C_B and C_D and 1.176 to 1.973 for C_W.

### `#1078` controls

Sections *Accuracy controls from #1078* and *#1078: controls where S fails and the variant has a smaller eta than S*. S
fails 52 of the 120 controls in both versions, with `η/eps` up to 4.50e+15; in the rows the section lists, its failures
lie at `ρ = eps/8` and, for the rotated designs among others, at `ρ = eps` and `32 eps` too. Each variant fails more
controls than S (68, 91 and 101). Among the 52 controls that S fails, the variant has a smaller `η` than S in 34 (C_B),
30 (C_D) and 30 (C_W), which does not by itself make it pass; in the rows the section lists, C_B passes two controls
that S fails (`rot1x2/neg/eps8` and `rot2x4z/neg/eps8`, Float64, 1.10.13). On the reproducer itself
(`c1078/f64/axis1x2/pos/eps8`) S and each variant fail in the listed 1.10.13 row; the 1.13.4 row is not listed. `#1078`
stays open.

### Raw results

The checksum list `sha256-list.tsv` covers the 31 gathered files that verdict mode read. The raw results (the gathered
campaign directory with its checksum list) are kept with the study archive.
