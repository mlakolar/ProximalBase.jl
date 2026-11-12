# Issue #865 evidence: shared numerical kernels

Environment: Linux x86_64, Julia 1.13.3, one Julia thread, BLAS one thread,
process pinned to one otherwise idle core (`taskset -c 12`). Baseline:
`master` at 1c2eb68a. Candidate: the tip of the #865 stack (dead-code removal,
kernels, consumer migrations).

## Bitwise characterization (`characterize.jl`)

```
julia --project=. docs/audit/numerical-kernels-865/characterize.jl BASELINE CANDIDATE
```

The script loads both source trees as modules in one process. It runs 148,528
oracle calls on each, on the same grid of ordinary and exceptional inputs, and
compares every result bit for bit. NaN payloads are ignored, and exceptions are
compared by type. The grid covers:

- the exponent range of Float32 and Float64, from the smallest subnormal to the
  top binade;
- SqrNormL2 value, gradient, value_and_gradient, prox!, prox_block and Scaled
  value parts. Arguments are dense, sparse, block and symmetric. Centers are
  none, same type, opposite sign (so `x - c` overflows), nextfloat
  (cancellation), Float64 at a Float32 argument, BigFloat, and entries that
  leave the range on rounding;
- LeastSquaresLoss value, gradient and value_and_gradient on dense, sparse and
  multi-response arguments, including overflowing `Y - Xβ`, overflowing partial
  products of `Xβ`, exact fits, and NaN/Inf data;
- LinearFunction value, NormL1 value/prox!/prox_block, and `norm_diff`.

| Branch vs master | Results | Differences |
|---|---:|---|
| `refactor/dead-code-865` | 148,528 | 0 |
| `refactor/numerical-kernels-865` | 148,528 | 0 |
| `refactor/numerical-utilities-865` | 148,528 | 112 results (194 leaves), all `NaN → ±Inf` |

The only difference is the SqrNormL2 gradient at an **infinite level** where the
residual is a nonzero subnormal. Master's cold path halved the residual, which
rounded it to zero, and returned `Inf·0 = NaN`. The new code halves only when
`x - c` actually overflows, so it returns the IEEE result `λ(x - c) = ±Inf`.
Example: `gradient(SqrNormL2(Inf, [0.0]), [-5e-324])` gives `[NaN]` on master
and `[-Inf]` now. This is a behavior change, documented in `numerics.md`, and
needs maintainer acceptance.

**Coverage check.** Each kernel was mutated in a scratch copy and the harness
rerun. All of these were detected:

- no halving at all;
- no halving in the scaled residual;
- `_exceptional_product` replaced by the ordinary product;
- the squared-residual exponent off by one;
- the R1 trigger always true;
- the least-squares half exponent off by one;
- the finite-residual least-squares recovery disabled;
- wide centers not rounded;
- the ridge gradient halving removed.

Two mutations cannot change any result, so they are covered only by
`test/numerical_kernels.jl`. One is the exponent attached to a zero significand.
The other is the slack margin of `_product_sum_shift`, which the unit test
checks against its bound directly.

## Paired benchmarks (`bench.jl`)

```
taskset -c 12 julia --project=benchmark --threads=1 -O2 --check-bounds=no \
  docs/audit/numerical-kernels-865/bench.jl BASELINE CANDIDATE 9
```

Both trees are loaded in one process and **share the same data arrays**. Only
function objects and iterates are built per tree. An earlier version allocated
the arrays per tree: then streaming loops whose code is identical in both trees
differed by up to 1.8x, from the relative placement of the input and output
arrays alone. Rounds alternate the order (A B, B A, …). Each measurement is
the minimum over a BenchmarkTools sample. The table reports the median over 9
rounds, the median and range of the per-round candidate/baseline ratios, and
the bytes allocated per call after warm-up. An A/A run (master against itself)
calibrates the harness noise.

Summary (stack vs master, median of per-round ratios; the A/A column is master against itself):

- **Ordinary paths:** every median is within ±2%, except two cases whose A/A ranges are as wide: Float64 `ls/value_and_gradient` 1.040 (A/A range 0.94–1.06) and Float32 `prox_block/centered` 0.950 (A/A 0.926). No allocation changes.
- **Exceptional paths:**
  - faster: the NormL1 scaled-product prox (0.72–0.74), the least-squares overflowing-Xβ recovery (0.88–0.92), the overflowing ridge gradient (0.92–0.96);
  - unchanged: the overflowing/underflowing squares and the value parts (1.000);
  - every residual overflowing: +0.6% (Float64) and +1.1% (Float32);
  - Float64 least-squares overflowing squares: +4.0% [1.012, 1.048] with A/A 1.004. That is below the 5% threshold but may be real; Float32 is 0.998. Earlier runs gave 1.008 and 1.020.
- Code placement moves the tight recovery loops by up to ±6% even with identical code (Float32 overflowing squares: A/A 1.063). Single ratios near 1.06 are within that band.

**Ordinary paths**

| Type | Case | master | stack | ratio (median) | range over rounds | bytes/call master → stack | A/A ratio [range] |
|---|---|---:|---:|---:|---|---|---|
| Float64 | `ridge/value/dense` | 1.38 µs | 1.38 µs | 1.001 | [0.996, 1.009] | 0 → 0 | 1.001 [0.995, 1.007] |
| Float64 | `ridge/value/centered` | 1.80 µs | 1.80 µs | 1.000 | [0.999, 1.002] | 0 → 0 | 0.999 [0.998, 1.002] |
| Float64 | `ridge/value/sparse` | 126 ns | 126 ns | 0.992 | [0.969, 1.008] | 0 → 0 | 1.000 [0.977, 1.008] |
| Float64 | `ridge/value/symmetric` | 4.00 µs | 4.00 µs | 1.000 | [1.000, 1.003] | 0 → 0 | 1.000 [0.999, 1.000] |
| Float64 | `ridge/gradient/dense` | 3.14 µs | 3.12 µs | 1.001 | [0.847, 1.151] | 16 → 16 | 1.015 [1.008, 1.018] |
| Float64 | `ridge/gradient/centered` | 4.21 µs | 4.22 µs | 1.003 | [1.000, 1.014] | 32 → 32 | 1.002 [0.987, 1.020] |
| Float64 | `ridge/value_and_gradient/centered` | 6.01 µs | 6.00 µs | 0.999 | [0.997, 1.003] | 48 → 48 | 1.001 [0.995, 1.016] |
| Float64 | `ridge/prox/dense` | 2.23 µs | 2.22 µs | 0.992 | [0.972, 1.005] | 16 → 16 | 0.997 [0.995, 1.001] |
| Float64 | `ridge/prox/centered` | 7.62 µs | 7.61 µs | 0.998 | [0.995, 1.001] | 32 → 32 | 1.000 [0.999, 1.001] |
| Float64 | `ridge/prox_block/dense` | 7.77 µs | 7.77 µs | 1.000 | [1.000, 1.000] | 32 → 32 | 1.000 [1.000, 1.001] |
| Float64 | `ridge/prox_block/centered` | 12.71 µs | 13.28 µs | 1.019 | [0.916, 1.123] | 48 → 48 | 1.040 [0.907, 1.073] |
| Float64 | `ls/value` | 14.11 µs | 14.01 µs | 0.996 | [0.948, 1.030] | 0 → 0 | 1.003 [0.985, 1.007] |
| Float64 | `ls/gradient` | 28.13 µs | 27.30 µs | 0.974 | [0.914, 1.000] | 0 → 0 | 0.980 [0.931, 0.986] |
| Float64 | `ls/value_and_gradient` | 26.62 µs | 27.90 µs | 1.040 | [0.987, 1.107] | 16 → 16 | 0.999 [0.942, 1.055] |
| Float64 | `linear/value` | 1.59 µs | 1.59 µs | 0.999 | [0.997, 1.003] | 0 → 0 | 0.997 [0.992, 1.008] |
| Float64 | `l1/value` | 1.30 µs | 1.30 µs | 0.999 | [0.997, 1.008] | 0 → 0 | 0.999 [0.998, 1.012] |
| Float64 | `l1/prox/weighted` | 6.18 µs | 6.20 µs | 1.003 | [1.000, 1.029] | 32 → 32 | 1.001 [1.000, 1.002] |
| Float32 | `ridge/value/dense` | 885 ns | 884 ns | 0.999 | [0.992, 1.005] | 0 → 0 | 1.001 [0.992, 1.006] |
| Float32 | `ridge/value/centered` | 932 ns | 930 ns | 0.998 | [0.996, 1.003] | 0 → 0 | 1.000 [0.998, 1.000] |
| Float32 | `ridge/value/sparse` | 81 ns | 81 ns | 1.000 | [0.988, 1.000] | 0 → 0 | 1.000 [1.000, 1.037] |
| Float32 | `ridge/value/symmetric` | 3.99 µs | 3.99 µs | 1.000 | [0.997, 1.003] | 0 → 0 | 1.000 [0.999, 1.000] |
| Float32 | `ridge/gradient/dense` | 1.17 µs | 1.15 µs | 0.994 | [0.980, 1.005] | 16 → 16 | 1.003 [0.997, 1.006] |
| Float32 | `ridge/gradient/centered` | 1.39 µs | 1.39 µs | 1.002 | [0.991, 1.007] | 32 → 32 | 1.003 [0.997, 1.004] |
| Float32 | `ridge/value_and_gradient/centered` | 2.37 µs | 2.41 µs | 1.014 | [0.977, 1.082] | 48 → 48 | 0.987 [0.961, 1.021] |
| Float32 | `ridge/prox/dense` | 1.15 µs | 1.16 µs | 1.006 | [0.993, 1.015] | 16 → 16 | 0.977 [0.769, 0.997] |
| Float32 | `ridge/prox/centered` | 3.89 µs | 3.90 µs | 1.003 | [1.001, 1.003] | 32 → 32 | 1.001 [0.998, 1.002] |
| Float32 | `ridge/prox_block/dense` | 7.77 µs | 7.77 µs | 1.000 | [1.000, 1.001] | 32 → 32 | 1.000 [1.000, 1.000] |
| Float32 | `ridge/prox_block/centered` | 13.21 µs | 12.39 µs | 0.950 | [0.869, 1.056] | 48 → 48 | 0.926 [0.888, 1.083] |
| Float32 | `ridge/value/wide-center` | 4.66 µs | 4.67 µs | 0.999 | [0.991, 1.005] | 160 → 160 | 1.013 [0.955, 1.016] |
| Float32 | `ls/value` | 6.29 µs | 6.28 µs | 0.995 | [0.993, 1.003] | 0 → 0 | 0.996 [0.994, 1.010] |
| Float32 | `ls/gradient` | 15.06 µs | 15.05 µs | 0.999 | [0.998, 1.004] | 0 → 0 | 0.999 [0.995, 1.002] |
| Float32 | `ls/value_and_gradient` | 15.01 µs | 15.06 µs | 1.003 | [0.988, 1.019] | 16 → 16 | 1.001 [0.986, 1.012] |
| Float32 | `linear/value` | 811 ns | 815 ns | 0.999 | [0.990, 1.010] | 0 → 0 | 1.000 [0.995, 1.005] |
| Float32 | `l1/value` | 807 ns | 806 ns | 0.998 | [0.993, 1.002] | 0 → 0 | 0.999 [0.991, 1.006] |
| Float32 | `l1/prox/weighted` | 3.16 µs | 3.12 µs | 0.987 | [0.965, 0.992] | 32 → 32 | 1.002 [0.996, 1.027] |

**Exceptional (recovery) paths**

| Type | Case | master | stack | ratio (median) | range over rounds | bytes/call master → stack | A/A ratio [range] |
|---|---|---:|---:|---:|---|---|---|
| Float64 | `exc/ridge/value/overflowing-squares` | 41.38 µs | 41.36 µs | 1.000 | [0.997, 1.000] | 0 → 0 | 0.999 [0.998, 1.002] |
| Float64 | `exc/ridge/value/underflowing-squares` | 419.52 µs | 419.50 µs | 1.000 | [1.000, 1.000] | 0 → 0 | 1.000 [1.000, 1.000] |
| Float64 | `exc/ridge/value_parts/scaled` | 41.70 µs | 41.72 µs | 1.000 | [0.999, 1.001] | 64 → 64 | 1.000 [0.999, 1.001] |
| Float64 | `exc/ridge/gradient/overflowing-residual` | 90.31 µs | 86.63 µs | 0.959 | [0.958, 0.972] | 32 → 32 | 1.048 [0.985, 1.060] |
| Float64 | `exc/ridge/value/overflowing-residual` | 826.25 µs | 831.38 µs | 1.006 | [1.006, 1.006] | 0 → 0 | 0.998 [0.998, 0.998] |
| Float64 | `exc/ls/value/overflowing-squares` | 16.77 µs | 17.45 µs | 1.040 | [1.012, 1.048] | 0 → 0 | 1.004 [0.968, 1.008] |
| Float64 | `exc/ls/value/overflowing-Xβ` | 9.44 µs | 8.70 µs | 0.921 | [0.893, 0.931] | 8288 → 8288 | 1.002 [0.992, 1.008] |
| Float64 | `exc/linear/value/overflowing-dot` | 23.83 µs | 23.84 µs | 1.000 | [1.000, 1.001] | 0 → 0 | 1.002 [1.000, 1.004] |
| Float64 | `exc/l1/prox/scaled-product` | 301.40 µs | 222.54 µs | 0.738 | [0.738, 0.740] | 48 → 48 | 1.003 [1.001, 1.004] |
| Float32 | `exc/ridge/value/overflowing-squares` | 43.27 µs | 43.27 µs | 1.000 | [0.999, 1.001] | 0 → 0 | 1.063 [1.063, 1.064] |
| Float32 | `exc/ridge/value/underflowing-squares` | 282.59 µs | 282.59 µs | 1.000 | [1.000, 1.000] | 0 → 0 | 1.009 [1.009, 1.009] |
| Float32 | `exc/ridge/value_parts/scaled` | 43.59 µs | 43.61 µs | 1.000 | [1.000, 1.001] | 48 → 48 | 1.061 [1.060, 1.061] |
| Float32 | `exc/ridge/gradient/overflowing-residual` | 82.61 µs | 75.96 µs | 0.920 | [0.918, 0.936] | 32 → 32 | 0.999 [0.974, 1.046] |
| Float32 | `exc/ridge/value/overflowing-residual` | 824.08 µs | 833.06 µs | 1.011 | [1.011, 1.011] | 0 → 0 | 0.998 [0.998, 0.998] |
| Float32 | `exc/ls/value/overflowing-squares` | 9.74 µs | 9.74 µs | 0.998 | [0.992, 1.003] | 0 → 0 | 1.002 [0.993, 1.006] |
| Float32 | `exc/ls/value/overflowing-Xβ` | 8.78 µs | 7.74 µs | 0.882 | [0.881, 0.883] | 4208 → 4208 | 1.000 [1.000, 1.002] |
| Float32 | `exc/linear/value/overflowing-dot` | 12.82 µs | 12.82 µs | 1.000 | [0.998, 1.005] | 0 → 0 | 1.001 [0.999, 1.003] |
| Float32 | `exc/l1/prox/scaled-product` | 283.21 µs | 205.02 µs | 0.724 | [0.723, 0.726] | 48 → 48 | 1.016 [1.015, 1.017] |


## Investigated performance findings

- **The halved residual as a select (fixed in #869).** The first version of
  the scaled-residual step used the `(d, h)` pair of `_halved_difference`. LLVM
  turned it into a branchless select, which computed `a/2 - b/2` and both
  scalings for every entry. The sum-of-squares recovery pass was 1.5x slower
  than master: 40.1 vs 26.8 µs for 10^4 entries. Moving the halved case out of
  line fixed that, but cost +5% when every residual overflows (one call per
  entry: 825 → 866 µs). The final form branches on the difference itself, as
  master does, with the halving rule in one inline helper `_halved`. Measured
  one tree per process: both recovery passes equal master (15.7 / 26.8 µs), and
  the all-overflowing value is within 1% (826 → 831 µs).
- **Placement artifacts (harness).** See above. Code paths this work does not
  touch (dense ridge `prox!`, the dense gradient loop) showed ratios between
  0.55 and 1.77 until the data arrays were shared.
