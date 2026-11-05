# Sphere prox of the ℓ1 norm, before and after #441 and #459

`sphere-l1-459.csv` holds the rows of `benchmark/sphere_l1.jl` for five checkouts, by `label`:

| label | commit | |
|---|---|---|
| pre | 42fe245 | before #441 |
| head | 1cf54ad | #441's head |
| master | 9dba71e | master when #459 started |
| step1 | a8b71e9 | #459, the one-subtraction margin comparison only |
| opt | da59ddb | #459, plus the SIMD sum of squares of a dense layout |

Apple M3 Pro, macOS, Julia 1.13.3, one thread, BenchmarkTools 0.4 s per case, one process per
checkout, run one after another. Times are medians in ns; `allocs` and `bytes` per call.
`pos` has margins above the thresholds (the shrink), `fb` none (the fallback); `extreme` puts
entries and thresholds near the bottom of the range (exceptional threshold products and the
mantissa/exponent scaling). Aliased rows copy the argument in the setup, outside the timing;
the 2 allocations of aliased `SparseIterate` `prox!` rows are the same in every checkout.

Reproduce, for an environment that develops a checkout and has BenchmarkTools:

    JULIA_NUM_THREADS=1 julia --project=<env> benchmark/sphere_l1.jl <label> <out.csv>
