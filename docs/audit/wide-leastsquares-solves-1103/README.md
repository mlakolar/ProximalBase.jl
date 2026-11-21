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

Measured package source: the tree of `fdf9b52777b280750a4402776ffeb94b7c702be6`. The study branch changes only this
folder, so `src/` at its head is that tree; `PB_SOURCE` can name a separate checkout of the commit instead.

## Files

| File | Role |
|---|---|
| `generate.jl` | writes the inputs as little-endian binary arrays with a manifest `inputs-index.tsv` (case, element type, shape, SHA-256 of each file) |
| `harness.jl` | the paired timing harness, the ADMM runs and the dump of the prox outputs |
| `accuracy.jl` | the BigFloat evaluation of each dumped prox output from the stored inputs |
| `summarize.py` | rebuilds every table and applies the criteria from the raw rows (Python standard library only) |
| `common.jl` | shared: the package modules, the prototypes, the roles, the manifest reader, the ADMM loop |

The scripts use only Julia standard libraries and run with `--project=.` at the package root. Raw output goes to
directories outside the repository; no data or output is committed.

## Roles

All roles are loaded in one process. Prototype modules and the package source are included twice, into modules
with different names, so that the A/A copies are compiled separately.

- **S**: the current `LeastSquaresLoss` prox. **S'**: a second include of the same source (its A/A control).
- **C_B**: the fat branch of Boyd's `lasso.m`. It caches `X'y` and the Cholesky factor of `I + XX'/ρ` and computes
  `q = X'y + ρv`, `β = q/ρ - X'((I + XX'/ρ) \ Xq)/ρ²`.
- **C_D**: the residual form. It caches the factor of `I + γXX'` and computes `a = (I + γXX') \ (y - Xv)`,
  `β = v + γX'a`; with `β = v + X'a'`, `(X'X + ρI)β = X'y + ρv` reduces to `X'((XX' + ρI)a' - (y - Xv)) = 0`.
- **C_W**: the whitened form. With `LL' = I + γXX'`, `W = L⁻¹X` and `c = L⁻¹y` it computes `β = v + γW'(c - Wv)`.
- **C_B', C_D', C_W'**: second copies of each prototype module (their A/A controls).

`ρ = inv(T(γ))` as in `src/functions/leastsquares.jl`, with `T` the element type of `X` and `y`. The prototypes check
`all(isfinite, X)` with the error text of the SVD path. Setup caches `XX'`; `refactor!(c, γ)` re-shifts and refactors
from the cached `XX'` (C_W also recomputes `W` and `c`). The solves use BLAS and LAPACK calls on preallocated
buffers, and the 4-argument `prox!` checks that its step is the one of the current factorization. The warmed `prox!`
of each prototype allocates 0 bytes in the check run of `harness.jl` for both element types.

The prototypes are not tuned: a production version could fuse vector passes. The ratios belong to these prototypes.

## Inputs

`generate.jl` writes dense Float64 and Float32 cases with `n < p`, and `X` and `y` of the same element type.

- Aspect ratios `n × p`: 100×1000, 500×5000, 1000×2000, 1000×1100 (near-square), 200×20000, 1500×5000.
- Designs, each with unit-norm columns: Gaussian; rank `n/2` (a product of Gaussian factors); duplicated rows (the
  first `n/2` rows repeated, so that `XX'` is singular). The rank-deficient designs have a consistent response
  `y = Xβ*` and an inconsistent one `y = Xβ* + noise`; the Gaussian design has `y = Xβ* + noise`. `β*` has about 100
  `N(0, 1)` nonzeros (density `100/p`) and the noise has variance `1e-3`, as in the wide lasso of Boyd et al., §11.1.
- Steps: `γ ∈ {1e-3, 1, 1e3, 1e6}` for Float64 and `{1e-3, 1, 1e3}` for Float32, that is `ρ = 1e3, 1, 1e-3, 1e-6`.
- Proximal inputs: eight columns per case. Four random `N(0, 1)` vectors are cycled in the timing loops, so no two
  consecutive calls repeat an input. Four `z - u` iterates come from an S-ADMM run on the same data (`ρ = 1`,
  `λ = 0.1‖X'y‖∞`), after 1, 5, 20 and 100 iterations; they are inputs a solver reaches.
- The wide lasso instances of §11.1 are the Float64 Gaussian cases 1500×5000, 500×5000 and 200×20000. The timing groups
  use the Gaussian cases and the consistent responses of the rank-deficient designs; their inconsistent responses are
  used for the accuracy evaluation.
- Accuracy controls: the reproducer of #1078 (`X = [1 0]`, `y = [1]`, `v = [±1/ρ, 0]`, `ρ = eps(T)/8`) and neighbours:
  `ρ ∈ {eps/8, eps, 32 eps}`, both signs, null-space components in `v`, 1×3 and 2×4 coordinate and diagonal designs
  (one with a zero singular direction), rotated 1×2 and 2×4 designs (full and deficient rank), a square and a tall
  design, in Float64 and Float32. They are used for accuracy only.

The streams of a Random generator can differ between Julia versions, so `generate.jl` runs once and every process reads
the files. `harness.jl` and `accuracy.jl` check each file against its SHA-256 in the manifest and stop on a mismatch.

## Measurements

One process writes one tab-separated file `timings-<julia>-<order>-<proc>-<groups>.tsv`: a header of `# key: value` lines
(Julia version, BLAS configuration, CPU, load average at start and end, the SHAs of the source and the harness, the
SHA-256 of the manifest) and one row per (case, measurement, parameter, round, role). The roles are timed in a fixed
order per process (`PB_ORDER`: S and S' before the C roles, or after them); the order within a round alternates.
Setup, array allocation and compilation follow the same order, and a warm-up for each element type compiles every role
before the first timed sample.

1. **First use** (`first_use`): construction, factorization and first prox on a fresh object at `γ = 1`, one sample per
   role and round, `GC.gc()` before each sample outside the timed region, with GC time, bytes and allocations from
   `@timed`. The time of each setup step of the primary roles is recorded too (`first_use_part`): the copy with the
   finiteness check, `svd`, the materialization of `V` and `U'y` for S; `XX'`, the factorization, `trsm` and `X'y` for
   the C roles.
2. **Warmed prox** (`warm`): for each step and each kind of input (random, ADMM), the minimum over `PB_BATCHES` batches
   of calls per role and round. A batch has at least 30 calls when one call takes under 0.1 s.
3. **Penalty change** (`penalty`): S calls with a new `γ`; each C role refactors from the cached `XX'` and then calls the
   prox. Two patterns: alternating between `γ = 1` and `1e3` (`alt2`), and a new step on every call from a grid of 16
   steps between `1e-3` and `1e3` (`newstep`).
4. **Warmed allocations** (`alloc`): `warmed_bytes(op::F) where {F} = (op(); op(); @allocated op())` behind a function
   barrier, for the prox and for the refactor-and-prox, for every role and case, with the reachable bytes of each object.
5. **Complete ADMM runs** (`admm_reference`, `admm_untimed`, `admm`), Float64, on the three lasso instances at
   `ρ ∈ {0.1, 1, 10}`. One loop serves all roles; only the x-update differs: `prox!(x, obj, z - u, 1/ρ)`. It follows
   Boyd's lasso loop with `α = 1`, `λ = 0.1‖X'y‖∞`, `z = soft(x + u, λ/ρ)`, `u += x - z` and the stopping rule with
   `ε_abs = 1e-4`, `ε_rel = 1e-2`.
   - `admm_reference`: S-ADMM at `ρ = 1` to the lasso stationarity `‖z - soft(z - X'(Xz - y), λ)‖∞ ≤ 1e-12 max(1, ‖X'y‖∞)`,
     checked every 10 iterations, at most `PB_REFERENCE_CAP` iterations; it stores the reference objective.
   - `admm_untimed`: for each role, the iteration count at which Boyd's rule is met and the count at which the stationarity
     is at most `1e-8 max(1, ‖X'y‖∞)` (checked every iteration, so its two matrix-vector products stay out of the timed
     runs), with the trajectory at checkpoints. A run that reaches `PB_ADMM_CAP` iterations without meeting the target
     is a lower bound and counts as inconclusive; its timed runs are skipped.
   - `admm`: timed runs of those counts with Boyd's `O(p)` residuals only, `PB_ADMM_REPEATS` repeats per process in
     alternating role order, with the A/A copies. Reported per run: iterations, factorization time (construction and
     factorization), iteration time, total time, the objective `½‖Az - b‖² + λ‖z‖₁`, its gap to the reference
     objective, and the stationarity at the end.
6. **Prox outputs** (`outputs`): each primary role's output for every case, step and input, from one untimed call, with
   the A/A copy's output compared bitwise; written as binary files with SHA-256s for `accuracy.jl`.

## Accuracy

`accuracy.jl` evaluates, from `X`, `y`, `v` and `γ` as stored and independently of the implementations:

- `η = ‖X'(Xβ - y) + ρ(β - v)‖∞ / (‖X'‖∞‖Xβ‖∞ + ‖X'y‖∞ + ρ(‖β‖∞ + ‖v‖∞))` in 256-bit BigFloat, with `ρ = 1/big(T(γ))`;
- the forward error for `n ≤ 500` against the 128-bit BigFloat Cholesky solution of the dual system
  `(XX' + ρI)a* = y - Xv`, `β* = v + X'a*` (a diagnostic).

An output fails accuracy when `η > 1e3 eps(T)` or when its role reported an error. The table of `η` reports `η_C/η_S`
beside pass and fail. Accuracy failures are tabulated apart from timing, and a cell whose variant fails accuracy at its
step has its timing reported and is left out of the speed criteria. A check pass before the campaign confirms that S
meets the threshold with at least a factor 10 in every main case; the threshold is not changed after the fact.

## Conditions

The campaign runs on exclusive nodes of one CPU kind, Julia 1.13.4 and 1.10.13 with `--threads=1 --check-bounds=no`
and one BLAS thread; an 8-thread run (`PB_BLAS_THREADS=8`) is a diagnostic. For each Julia version: three fresh
processes per role order, 21 rounds each. The package source, harness and generator SHAs are passed in `PB_SOURCE_SHA` and
`PB_HARNESS_SHA` and recorded in every output file. The working set of each object (`workingset`) and the L3 size
(`l3_cache` in the header) are recorded for comparison.

## Criteria

Pre-registered in the plan; the summarizer applies them as follows.

- Per cell (case, measurement, parameter) and variant `V ∈ {C_B, C_D, C_W}`, each process gives the per-round paired
  ratios `V/S`, `S'/S` and `V'/V` and their medians (rounds are paired within the process).
- **Faster**: all per-process medians of `V/S` are below 0.95 and the distance of the largest median from 1 exceeds the
  A/A spread, the largest `|median - 1|` of `S'/S` and `V'/V` over the processes of the cell. **Slower** is the
  symmetric condition above 1.05. Anything else is **inconclusive**, not equal. The plan asks for six processes (three
  per role order); a cell with fewer is marked partial.
- Recommendation per variant, on every Julia version in the data, for Float64 and one BLAS thread:
  - **(A)** propose a PB fixed-penalty backend as a follow-up design issue: `V` passes `η` in every case (steps and
    inputs, Float32, rank-deficient, near-square and the #1078 controls included) where S passes; it is not slower on
    the warmed prox in any primary cell; its whole-ADMM total is faster on §11.1 (1500×5000) at `ρ = 1` at both stopping
    levels, with the objective gap at most that of S plus `1e-8 max(1, |p*|)`.
  - **(B)** solver-side factorization only, when `V` fails (A) but passes `η` at the §11.1 inputs at step 1, is faster in
    whole-ADMM total at both levels and is not slower at the tight target in any instance and `ρ`.
  - **(C)** no change otherwise.
  - In every outcome #1078 is unaffected; the summarizer tabulates the control residuals of each variant against S.
- Cases, tolerances and criteria do not change after the results are seen.

## Rerun

Run from the package root with `julia --threads=1 --check-bounds=no --project=.` (Julia 1.13.4 for the generator). Replace
`<inputs dir>`, `<output dir>` and `<source dir>`; `PB_SOURCE` can be omitted when `src/` of this checkout is the
measured tree.

1. Smoke checks (seconds to minutes; also exercise the check of the prototypes against S):

   ```
   PB_CHECK_ONLY=1 PB_INPUTS=<small inputs dir> julia ... docs/audit/wide-leastsquares-solves-1103/generate.jl
   PB_CHECK_ONLY=1 PB_INPUTS=<small inputs dir> PB_OUTPUT=<small output dir> julia ... docs/audit/wide-leastsquares-solves-1103/harness.jl
   PB_CHECK_ONLY=1 PB_INPUTS=<small inputs dir> PB_OUTPUTS=<small output dir>/outputs-<julia> julia ... docs/audit/wide-leastsquares-solves-1103/accuracy.jl
   ```

2. Inputs, once: `PB_INPUTS=<inputs dir> julia ... generate.jl` (`PB_PROFILE=small` for the small profile,
   `PB_FILTER=<regex>` for a subset of the cases).
3. Reference optimum and iteration counts, once per Julia version (`PB_FILTER` selects an instance, for example `1500x5000`):

   ```
   PB_GROUP=admm_reference PB_INPUTS=<inputs dir> PB_OUTPUT=<output dir> PB_REFERENCE=<reference dir> julia ... harness.jl
   PB_GROUP=admm_untimed   PB_INPUTS=<inputs dir> PB_OUTPUT=<output dir> PB_REFERENCE=<reference dir> julia ... harness.jl
   ```

4. Timed processes, each a fresh process, for `PB_ORDER` in `S_first`, `C_first` and `PB_PROC` in 1, 2, 3 (the groups can be
   split over processes, for example `PB_GROUP=first_use,warm` or `PB_GROUP=admm`):

   ```
   PB_ORDER=S_first PB_PROC=1 PB_ROUNDS=21 PB_GROUP=first_use,warm,penalty,alloc,admm PB_SOURCE_SHA=<sha> PB_HARNESS_SHA=<sha> \
   PB_INPUTS=<inputs dir> PB_OUTPUT=<output dir> PB_REFERENCE=<reference dir> julia ... harness.jl
   ```

5. Prox outputs and accuracy, per Julia version:

   ```
   PB_GROUP=outputs PB_INPUTS=<inputs dir> PB_OUTPUT=<output dir> julia ... harness.jl
   PB_INPUTS=<inputs dir> PB_OUTPUTS=<output dir>/outputs-<julia> julia ... accuracy.jl
   ```

6. Summary of one or more output directories gathered into `<summary dir>` (the output directories of all processes):

   ```
   python3 -I docs/audit/wide-leastsquares-solves-1103/summarize.py <summary dir>
   ```

   It prints markdown tables and writes `tables.md` and `verdict.json` into `<summary dir>`.

Environment variables of `harness.jl`: `PB_INPUTS`, `PB_OUTPUT` (required unless `PB_CHECK_ONLY=1`), `PB_REFERENCE`
(directory of the reference optimum and the iteration counts; default `PB_OUTPUT`), `PB_ORDER`, `PB_PROC`, `PB_ROUNDS`
(default 21), `PB_BATCHES` (3), `PB_GROUP` (comma-separated subset of `first_use`, `warm`, `penalty`, `alloc`, `outputs`,
`admm_reference`, `admm_untimed`, `admm`), `PB_FILTER` (regular expression on case ids), `PB_BLAS_THREADS` (1),
`PB_ADMM_REPEATS` (5), `PB_ADMM_CAP` (20000), `PB_REFERENCE_CAP` (100000), `PB_SOURCE`, `PB_SOURCE_SHA`,
`PB_HARNESS_SHA`, `PB_CHECK_ONLY`.

## Results

Not yet available: the campaign has not run. After it, the tables of `summarize.py` and the recommendation are added here.
