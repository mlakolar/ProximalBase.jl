# Local block-prox performance comparison

Fresh source loading (`--compiled-modules=no`), explicit `--check-bounds=no`,
one Julia/BLAS thread and CPU 2. Four repetitions per source/runtime; repetition
1 warms the complete case set, and the table reports median paired ratios of
repetitions 2–4. Ratio >1 means the candidate took longer.

All scalar/group/public-writer hot calls allocated zero bytes. Complete solves
had identical allocation counts and pass counts; objectives agreed in argument
precision. Numerical tests were separately run with bounds checks enabled.

| Whole solve | Julia 1.10.13 | Julia 1.13.3 |
| --- | ---: | ---: |
| Float32/scalar | 0.999× | 0.799× |
| Float32/group_ls | 1.049× | 1.016× |
| Float32/group_quad | 0.988× | 1.031× |
| Float32/group_ls_csc | 1.002× | 0.968× |
| Float32/generic_linf | 1.032× | 1.022× |
| Float32/generic_sparsegroup | 1.036× | 1.012× |
| Float32/multiresponse | 1.015× | 1.028× |
| Float64/scalar | 0.999× | 0.999× |
| Float64/group_ls | 0.985× | 1.027× |
| Float64/group_quad | 1.056× | 1.025× |
| Float64/group_ls_csc | 1.000× | 0.923× |
| Float64/generic_linf | 0.992× | 1.003× |
| Float64/generic_sparsegroup | 0.997× | 1.004× |
| Float64/multiresponse | 1.007× | 1.008× |

The largest whole-solve median ratio was 1.056×. These are local exploratory measurements, not confidence bounds.

The largest small-block writer ratio was 1.417×: `group_write/Float32/GroupNormL2/n32/zerofalse` on Julia 1.13.3. Median runtimes were 48.0 → 69.0 ns (+21.0 ns); the ratio is the median of per-repetition ratios. This is the measured cost of the separate numerical buffer and public write, and requires a maintainer tradeoff decision before merging. The generalized bulk-writer redesign remains #787.

The `zero=true` case labels use small trial entries of 0.01; large Linf groups
can remain nonzero. Literal-zero behavior and KKT zero-group skips are covered
by the numerical and solver tests.

Earlier timings were discarded: randomized CD order and cached code with
different bounds-check behavior made them unsuitable for an API comparison.
Fresh-source optimized Float64 least-squares/NormL1 scalar CD IR is identical after the approved rename.

[Raw measurements](measurements/) and [machine-readable results](performance.json) retain each repetition, allocation reading, convergence pass count and objective.
