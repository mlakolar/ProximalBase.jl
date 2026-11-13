# Common brief for the four #901 architecture implementers

You are one of four implementers working **in parallel** on competing architectures for the NormL2 /
GroupNormL2 ℓ2 kernels of ProximalBase.jl (PR #901, issue #900). An orchestrator compares the four
designs; you own exactly one. Your job: make your design as fast as possible while numerically sound,
and keep it simple. Escaping local optima is the point: do not converge on another implementer's design.

## Background (read first)

- PR #901: `gh pr view 901 --comments` (read the last ~6 comments: rounds 9–10, the maintainer decisions
  D-X / D-MPFR, and the 2026-11-13 20:28 campaign log). `gh issue view 900`. Read-only: never comment,
  review, push or open PRs.
- `docs/design/numerical-kernels-865.md`, `docs/src/numerics.md` (the contract), `benchmark/README.md`,
  `benchmark/l2_cold_paths.jl` (case definitions of the regressions).
- Trees: `base` = `/home/mkolar/projects/juliaPkg/pb901x/base` (76964085 + master linear.jl: the original
  separate fast paths in `src/functions/norml2.jl`), `ref` = `/home/mkolar/projects/juliaPkg/pb901x/ref`
  (0eda76a9: kernels consolidated into `src/numerical_kernels.jl`, shared with SqrNormL2, LinearFunction,
  LeastSquaresLoss, NormL1). `git diff 76964085 0eda76a9 -- src` is the consolidation.
  `/home/mkolar/projects/juliaPkg/pb901x/active` (04821b7e) is the active #901 agent's in-progress
  small-group repair: you may read it, never modify it.
- The measured regressions of `ref` vs `base` (local Raptor Lake, both role orders): GroupNormL2
  zero-weight `value` with groups of 5 and 32 up to **4.6×** (987 → 4,075 ns); groups of 2 and 5 up to
  **1.56×** across the overflow / underflow / threshold regimes; ordinary
  `Group/Float64/m1024/range/dense/prox` +1–4%. Groups of 1024 / 65,536 are 4–8% *faster* at ref.
  Likely cause: per-group overhead of the shared cold kernels.

## Rules

- Work only in your worktree / branch (given below). Commit there with clear messages; never push,
  never touch other worktrees, `feat/issue-900-groupnorml2`, `consolidate/issue-900-kernels`, master.
- Scope: `src/` (NormL2, GroupNormL2, and the shared kernels in `src/numerical_kernels.jl` they use),
  tests and comments as needed. If you change a shared kernel, its other callers (SqrNormL2,
  LinearFunction, LeastSquaresLoss, NormL1, norm_diff) must stay correct and not get slower.
  No new BigFloat-only recovery machinery (#366 scope policy, maintainer rulings on #901). Don't expand
  exceptional recovery; you may simplify it if the contract (numerics.md) is still met.
- Acceptance criteria and performance protocol: `/home/mkolar/projects/juliaPkg/pb901x/harness/experiment901/METHODOLOGY.md`.
  **Bitwise agreement with base is not required**; mathematical correctness, the documented contract,
  NaN/Inf semantics, zero weights, safe mutation and zero allocations are.
- **Local machine:** another agent is running a timing campaign on this machine. Run local Julia only
  pinned to your two E-cores (`taskset -c <your cores> julia ...`, cores below), one Julia process at a
  time, and **never time anything locally** (no BenchmarkTools locally, no performance conclusions from
  local runs). Use local runs for compiling, single test files, quick checks.
- **Discovery** (all timing, and test tiers if you like). Read `~/.claude/skills/slurm-workflow/SKILL.md`,
  `references/discovery.md`, `references/ssh.md`. Use the SSH session given below (never close it,
  never start another); at most **2 concurrent cluster commands** (count wait loops). Project dir
  `/scratch2/mkolar/pb-901x` (shared depot, seeded by the orchestrator); your files go under
  `/scratch2/mkolar/pb-901x/<your letter>/`. Account `mkolar_1314`; timing on partition `oneweek` per
  METHODOLOGY.md; tests may use `main` (titan_run.sh, tag `impl-<letter>`). Jobs ≤ 2 days. You may use as
  many nodes as is useful, but cheaply screen before anything large.
- **Common harness.** The orchestrator is writing `experiment901/screen901.jl` (paired screening timing,
  in-process base/A-A/ref/candidate), `experiment901/accuracy901.jl` (BigFloat accuracy + semantics) and a
  submit script, in `/home/mkolar/projects/juliaPkg/pb901x/harness/experiment901/`. You will be messaged
  when they are ready (until then, implement and test). Use them for every number you report so all four
  designs are measured identically. You may add design-specific diagnostic probes of your own, reported
  separately.
- Iterate: implement → tests → accuracy harness → screening on Discovery → profile/inspect native code
  (`code_llvm`/`code_native` locally is fine) → improve. Several revisions are expected. Stop when your
  design meets the gate (no screened case > 1.05× base in both orders beyond A/A; zero-weight and
  small-group regressions gone; m=1024 not worse than base) and further changes give no measurable gain,
  or when you hit a fundamental limit of the design (then say so with evidence).
- Do not merge or combine other implementers' code. If you see an idea from another design that would
  help yours, mention it in the report rather than copying it.

## Final report (your last message; the orchestrator relays it)

1. Commit SHAs (final and notable intermediate), branch.
2. Design summary: what is shared, what is separate, what was removed/added; why it is fast.
3. Correctness: test results (files, pass/fail counts, Julia version), accuracy-harness summary (max error
   in bound units per op/type/regime vs base), any changed tests with justification, allocation checks.
4. Performance: screening results from the common harness (per-case table of candidate/base ratios forward
   and reversed, A/A, ref/base for comparison; worst cases), Slurm job IDs, nodes, CPU type, wall times.
5. Complexity metrics (METHODOLOGY.md list).
6. Risks, limitations, ideas you'd suggest combining, and your honest assessment of the design.
