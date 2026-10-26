# Validation receipts

These are the final local execution receipts from 26 October 2026. Each JSON
records the command, commit, implementation hashes, elapsed time and exit code.
All Julia workloads ran sequentially with one Julia and one BLAS thread.

The complete 39-module suites ran at `f68cd73fd5efcc1b7a59e4c8099f17e7060cafa3`.
The only later implementation edit was the constrained-domain guard. Its five
related suites, including allocation checks and Aqua, ran on both versions at
`c0e13053a5b4d19d113f2d2815cd1c15563268e3`. CI reruns the complete suite on the
assembled PR head. Immutable final CI/head receipts accompany the delivery.

`baseline-probes.log` loads a recovered frozen copy of master f5c5797 rather
than the repair module. `mixed-precision-before.log` loads a frozen copy of
f68cd73 and retains the introduced guard rejection that the final edit repairs.
The source bundles, recreation scripts and dirty downstream provenance are
preserved in the workspace validation folder adjacent to the audit worktree.
No downstream source or patch was added to this package.

The first installed-user harness attempts failed: Julia 1.13 found a soft-scope
counter error in the harness, and Julia 1.10 rejected a worktree path as a Pkg
URL. The corrected harness installs the exact commit through an ordinary Git
clone, checks loading and public operations, and executes all four README
Julia examples. Both retry receipts passed. The original failures remain here.

The 75 Float64 Hypatia cases passed 1,206 assertions. Three first oracle solves
were rejected as ALMOST_OPTIMAL and retried using the suite's existing explicit
retry policy. Accepted solves had OPTIMAL/FEASIBLE_POINT statuses. Solver
warnings and rejected statuses remain in the log; tolerances were not relaxed
by this audit.

Warmed CPU diagnostics cover twelve paths in Float32 and Float64, with zero
measured bytes per call. Timings are one-process exploratory observations,
not a speedup comparison. CPU-backed JLArrays group diagnostics allocated
2,184,336 and 2,440,032 host bytes per call; those are not physical GPU device
allocation or timing measurements.

The link-enabled Documenter build executed doctests, cross-references, export
checks and all eighteen Literate examples. Its API page size warning and local
remote-HEAD warning are retained; deployment was skipped. The separate link
inventory covers maintained README/benchmark/optional-validation guidance too.
Its first check found a not-yet-deployed migration URL; the README now links to
the source guide. The final inventory records the repaired links.

Recreate the full suites with each supported Julia binary and
`scripts/test_master.jl`. Build with `DOCUMENTER_LINKCHECK=true julia
--project=docs docs/make.jl` after `julia scripts/instantiate.jl docs`. The oracle
recipe is `julia scripts/instantiate.jl test/oracles`, then `julia
--check-bounds=yes --project=test/oracles test/oracles/runtests.jl`.
