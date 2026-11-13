# #901 architecture experiment: common methodology

Orchestrator-owned. Implementers read it; they do not edit it (propose changes to the orchestrator).

## Trees and roles

| Label | What | Where (local) |
|---|---|---|
| `base` | performance baseline: `76964085` + master's `linear.jl` (commit `f36a6d6a`, branch `exp/901-baseline`) | `/home/mkolar/projects/juliaPkg/pb901x/base` |
| `ref` | frozen reference `0eda76a9` (#901 head at experiment start) | `/home/mkolar/projects/juliaPkg/pb901x/ref` |
| `active` | read-only: `04821b7e`, the active #901 agent's in-progress small-group repair (comparator only) | `/home/mkolar/projects/juliaPkg/pb901x/active` |
| `A`..`D` | candidates, branch `exp/901-<X>-...`, started from `0eda76a9` | `/home/mkolar/projects/juliaPkg/pb901x/<X>` |

Every measurement names trees by commit SHA. Never measure uncommitted code on Discovery.

## Numerical acceptance criteria (bitwise agreement with base is NOT required)

Supported types: Float32, Float64 (homogeneous). BigFloat: must still work and pass existing tests
where the tests express the mathematical contract; do not add BigFloat-specific recovery machinery.
Differences from base in BigFloat at MPFR exponent limits, or under directed rounding, are acceptable.

N1. **Test suite.** `julia scripts/test.jl` (quick tier = all files) passes on Julia 1.13.
   A test that asserts a *bit pattern or exact float* that the documented contract does not require may be
   changed to a tolerance or a BigFloat reference, with a one-line justification in the commit message.
   A test that encodes a documented semantic (NaN/Inf classes, +0, unchanged zero-weight groups,
   errors before writes, allocations) must not be weakened.
N2. **Accuracy vs independent high-precision reference** (common harness `experiment901/accuracy901.jl`,
   BigFloat at 512 bits, computed from the mathematical formulas with no package code):
   - values (finite exact result with |exact| ≤ floatmax/4):
     `|a − e| ≤ 2(n+4)·eps(T)·|e| + 4K·nextfloat(T(0))`, n = entries touched, K = groups;
   - prox entries: `|y_i − y*_i| ≤ (m+8)·eps(T)·|x_i| + 4·nextfloat(T(0))`, m = group size
     (backward-stable bound; it covers cancellation near the threshold), and a group whose exact norm is
     at most `(1 − 4(m+4)eps)` times its threshold must be exactly zero (+0);
   - screen (`kkt_violation`): `|a − e| ≤ (m+8)·eps(T)·(‖g_G‖ + λw) + 4·nextfloat(T(0))`;
   - additionally, per (op, type, regime), the candidate's max error in units of the bound must not exceed
     max(2 × base's, base's + 0.25 of the bound): no material accuracy regression even when inside the bound.
   Regimes: ordinary; wide dynamic range; sum of squares overflows; underflows (incl. subnormal entries);
   threshold `γλw` overflowing/underflowing with representable result; values ≥ floatmax/2 representable;
   group sums < K·floatmin.
N3. **Semantics, exact** (checked by tests and by the accuracy harness's class checks): NaN/±Inf result
   classes of `docs/src/numerics.md`; zero-weight groups returned bit-identical (`===` per entry) and
   contributing 0; infinite weight / infinite level ⇒ `+0`; `γ = 0` or `λ0 = 0` ⇒ identity; negative/NaN
   step throws before writes; `prox!(x, f, x, γ)` works; `value_and_prox!` returns value at stored output.
N4. **Allocations.** `prox!`, `value`, `value_and_prox!`, `prox_block!`, `kkt_violation` allocate 0 bytes
   after warm-up for Float32/Float64 dense, BlockIterate and SparseIterate where they do today.
N5. **Inference.** Existing `@inferred` tests pass.

## Performance protocol

- Discovery, partition `oneweek`, account `mkolar_1314`. One hardware type per comparison: either
  `--constraint=xeon-4116` (Skylake, 24 cores) or `--constraint=xeon-2640v4` (Broadwell, 20 cores);
  report which. Never compare ratios across hardware types; compare candidates only via ratios to `base`
  measured in the same process/node.
- `--exclusive` node. Julia 1.13.3 (`~/julia/julia-1.13.3/bin/julia`), `--threads=1 --check-bounds=no
  --startup-file=no`, `taskset -c <core>`, BLAS threads 1.
- In-process paired design: the harness compiles `base` twice (X and the A/A control XC), plus the
  tree(s) under test, as separate modules; inputs/outputs shared across roles; every round times all roles
  in rotated order. Each campaign is run **forward** (X = base) and **reversed** (X = candidate) in
  **fresh processes**. A case counts as slower only if slower in both orders and beyond the A/A spread.
- Concurrency on an exclusive node: at most one benchmark process per physical core, cores spaced
  (default: 6 concurrent processes on a 24-core 4116 node, pinned to cores 0,4,8,12,16,20; 5 on a 2640v4
  node, cores 0,4,8,12,16). Calibration (orchestrator) decides whether this is safe vs 1 process/node.
- Metrics, per case: median ns/call of each role; paired per-round ratio candidate/base (median over
  rounds); A/A ratio; allocations (bytes) per call. Per campaign: geomean by category, worst case, number
  of cases > 1.05 and > 1.02 in both orders, the A/A distribution.
- Gate (from the maintainer): no case reproducibly > 5% slower than `base` without explicit approval;
  accept A/A-level noise. Primary targets: GroupNormL2 zero-weight values (4.6× at ref), small groups
  m = 1, 2, 5 (up to 1.56× at ref), m = 32, ordinary Float64 m = 1024 prox (~1–4% at ref); keep the
  m = 1024/65536 cold-path gains of ref.
- Screening first (`experiment901/screen901.jl`, small curated case set). Full campaign
  (`benchmark/l2_cold_paths.jl` 5,046 cases + `benchmark/l2_norms.jl` + consumer harnesses) only for the
  strongest designs, run by the orchestrator.

## Complexity metrics (reported by each implementer)

`git diff --stat 0eda76a9..HEAD -- src` and `76964085..HEAD -- src`; lines of `src/numerical_kernels.jl`,
`src/functions/norml2.jl`, `src/functions/groupnorml2.jl`; count of `@noinline`/`@inline` kernel functions
serving NormL2/GroupNormL2; number of independent copies of a recovery algorithm (scaled norm, exponent
sum, threshold product) — the consolidation exists to have one of each.

## Calibration (orchestrator, 2026-11-13, Discovery oneweek, Xeon E5-2640 v4, jobs 14217165 and 14217167)

- One process on 23 screening cases: 7 s; the full 826-case screen with 3 roles and 11 rounds, split in 5
  shards: 34–38 s per shard. Screening is cheap: use all cases and several fresh processes.
- Concurrency on an exclusive 20-core node, same cases: base times rise by 7% (5 processes), 15% (10), 28%
  (20) (turbo frequency), but the paired ref/base ratio moves by about 1% at the median in every phase.
  Single cases move by 5–10% (up to 40%) **between processes**, also between solo runs: placement noise.
  Hence: **3 fresh processes per role order** and judge a case by all of them; up to 10 concurrent
  processes per 2640v4 node are acceptable for paired ratios.
- A/A per process: median |XC/X − 1| about 0.6%, p90 about 4.5%, max about 27% (single cases).
- ref/base on Discovery (fwd/rev): zero-weight `value` 2.0–3.1× (m = 1, 2, 5, 32), zero-weight fused
  1.2–1.3×, small-group overflow `kkt` 1.3–1.4×, NormL2 n = 1, 2 exceptional values 1.2–1.3×; NaN groups
  and m = 1024 cold paths faster. Overall geomean 0.935.
- Hardware for the whole experiment: `--constraint=xeon-2640v4` (Broadwell, AVX2, like the local AVX2
  machine; most free capacity in oneweek).

## How to run

Paths: `H=/home/mkolar/projects/juliaPkg/pb901x/harness/experiment901`. Commit first; trees are copied by SHA.

Accuracy (local is fine, pinned to your cores; no timing):

    cd <your scratch dir> && PB_TREES="base=/home/mkolar/projects/juliaPkg/pb901x/base;X=/home/mkolar/projects/juliaPkg/pb901x/X" \
      PB_OUTPUT=$PWD/acc-<sha> PB_TRIALS=6 taskset -c <cores> julia --threads=1 --startup-file=no \
      --project=/tmp/claude-1000/-home-mkolar-projects-juliaPkg-ProximalBase-jl/f8b9f9c7-b78d-4e7d-b03f-28871ee5aa1e/scratchpad/benchenv $H/accuracy901.jl

  (exit code 0 = all criteria met; it prints violations, class failures and regressions vs base.)

Screening (Discovery; standard = all 826 cases, 3 forward + 3 reversed fresh processes, 6 concurrent):

    $H/discovery/submit_screen.sh -s <your session> -n <X>-<short sha> -c <X>=/home/mkolar/projects/juliaPkg/pb901x/<X> \
       -o fwd1,fwd2,fwd3,rev1,rev2,rev3 -k 1 -p 6 -r 15
    # prints "job <id> run <remote dir>"; about 10 min once running. Wait with a bounded sacct loop.
    $H/discovery/fetch_run.sh <your session> <remote dir> <local dir>
    python3 -I $H/summarize901.py --num <X> --den base --also ref <local dir>/<run>/o*_s1 > summary.txt

  Focused re-screens: add `-f '<regex>'` (no commas, no single quotes) on case ids such as
  `Group/Float64/m5/range/w0/value`. Quote the ratios of the standard screen in reports.
  The order is fwd = base, ref, X; rev = X, base, ref (X is then the A/A role).
