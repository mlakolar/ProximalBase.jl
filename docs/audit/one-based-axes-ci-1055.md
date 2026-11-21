# CI runtime of the one-based axes tests

This investigation for [#1055](https://github.com/mlakolar/ProximalBase.jl/issues/1055) used baseline `4e26f7324ced50544d9ab89ee3d32dbca2c3e830`. The isolated test file is dominated by compilation. The two processes took about 0.28 seconds combined to start, and the two precision testsets accounted for less than one second of median wall time after subtracting their compilation counters.

## Reproduction

Use the package's normal selector with Julia 1.13.4.

```sh
julia --startup-file=no --threads=1 --project=. scripts/test.jl one_based_axes
```

`Pkg.test` constructs the test environment and the runner starts an isolated file process with `--check-bounds=yes`. The file starts a separate Julia process with `--check-bounds=auto` for its default-bounds cases. Use the `PB_TEST_FILE_TIME` record for the file-level endpoint. Each complete reproduction passed 3,246 assertions per precision and two assertions for the default-bounds child, totaling 6,494.

The one-off cluster (Slurm) canary took 298.944 seconds. Job 14433113 ran three stock and three instrumented invocations with alternating pair order on one bound AMD EPYC core, with one Julia and one BLAS thread and matching dependency manifests. The stock files took 262.279, 263.308, and 263.617 seconds. This 1.338-second range describes that allocation. Other allocations and hosted runners showed wider variation.

## Where the time went

The scratch profiler enabled `Base.cumulative_compile_timing(true)` before fixture loading and recorded `Base.cumulative_compile_time_ns()` deltas around imports and testsets. It measured wall time in each process and validated that the runnable intervals were contained in the corresponding parent-side launch and wait intervals. Child times are included in file wall time. The compilation column for the full file sums the counters from both processes.

| Interval | Median wall (s) | Wall range (s) | Median counted compilation (s) |
| --- | ---: | ---: | ---: |
| Instrumented isolated file | 266.359 | 265.826–284.534 | 258.955 |
| File process launch | 0.133 | 0.131–0.134 | — |
| Default-bounds child process launch | 0.145 | 0.144–0.145 | — |
| Default-bounds child testset | 8.769 | 8.681–27.186 | 7.418 in the child process |
| Float32 testset | 125.920 | 125.541–128.000 | 125.594 |
| Float64 testset | 125.471 | 123.309–125.759 | 124.900 |

The child import interval took 18.980 seconds in the first instrumented run and 0.486 and 0.407 seconds in the later runs. The combined compile counters varied from 258.571 to 259.258 seconds across all three runs. They accounted for 91.01% of the first instrumented run and 97.27–97.33% of the later two. The two later instrumented runs were 2.518 and 2.742 seconds longer than their adjacent stock runs. Residual time includes execution, imports, test framework work, GC, and waiting. The median combined residual inside the precision testsets was 0.919 seconds. About five seconds fell outside the per-precision markers within the enclosing loop.

| Largest named sections across both precisions | Median wall (s) | Median counted compilation (s) |
| --- | ---: | ---: |
| Offset input, destination and step array | 70.033 | 69.691 |
| One-based array parity | 59.031 | 58.975 |
| QuadraticFunction | 30.682 | 30.559 |
| SymmetricSparseIterate | 25.296 | 25.189 |
| LeastSquaresLoss with a matrix response | 20.070 | 20.047 |

See the [profiling record](https://github.com/mlakolar/ProximalBase.jl/pull/1140#issuecomment-7060036297) for process settings and section ranges.

## Optimization trials

The adoption criterion was a median paired file-time reduction greater than 5%, an absolute saving larger than the stock spread, and preservation of the existing tests on Julia 1.13.4 and 1.10.13.

Reducing specialization of the local rejection helper produced Julia 1.13.4 stock and candidate pairs of 280.316 → 275.396, 287.212 → 276.908, and 269.624 → 262.209 seconds. All six files passed. The median paired reduction was 7.415 seconds (2.75%) and stock times spanned 17.588 seconds. Julia 1.10.13 also passed all 6,494 assertions at both revisions. The trial missed the timing criteria. [Helper results](https://github.com/mlakolar/ProximalBase.jl/pull/1140#issuecomment-7060559398)

Capping optimization at level 1 inside the test module gave 280.214 seconds stock and 278.625 seconds with the cap, measured in a warm same-allocation pair on Julia 1.13.4 in job 14434153. Both preserved all 6,494 assertions. The 1.589-second saving (0.57%) fell below the 4% preliminary screen, which ended this trial after one pair.

## Hosted CI observations

The file ran in shard 1. These three runs used the same baseline tree, Julia 1.13.4, and restored job caches.

| Run | File (s) | Test steps 1 / 2 / 3 (s) | Longest shard |
| --- | ---: | --- | ---: |
| [Baseline PR](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44881286546) | 303.723 | 802 / 509 / 548 | 1 |
| [Baseline dispatch 1](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44882619503) | 220.607 | 574 / 607 / 696 | 3 |
| [Baseline dispatch 2](https://github.com/mlakolar/ProximalBase.jl/actions/runs/44882621579) | 283.849 | 754 / 573 / 700 | 1 |

The longest test step ranged from 696 to 802 seconds across these runs. A reduction in shard 1 can shorten the longest test step only while shard 1 determines that maximum. Both candidates stopped at the Slurm evaluation. The hosted observations characterize the baseline variation and identify the limiting shard in each run.
