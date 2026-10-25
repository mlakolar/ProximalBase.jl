# ProximalBase repeated CPU baseline

21,080 valid case/run measurements; 2,108 case/version groups. Target: 10 independent runs per case.

The primary time is the arithmetic mean of run means. Each run uses a fresh Julia process and a keyed seeded input draw. The standard deviation and Student-t interval use independent runs, not individual samples. Input variability and machine noise are both included. No adverse run is discarded.

Within each run, BenchmarkTools collects up to the configured sample/time limit. Compilation, validation and prepare callbacks are excluded. Cold cases include construction and the first proximal call. CPU profiles include prepare callbacks; allocation profiles isolate the measured operation. These CPU measurements do not cover GPU hardware or every parameter combination.

Validation errors: 0. Incomplete case/version groups: 0. Duplicate groups: 0.

## Unexpected warmed allocations

These are candidates for inspection, not automatically confirmed implementation defects.

| Case | Julia | Mean bytes | Maximum bytes | Runs |
|---|---|---:|---:|---:|
| None observed | | 0 | 0 | |

## Variable cases

CV is the between-run standard deviation divided by the mean. Large CV can reflect input-dependent work, short timing resolution, or shared-node noise.

| Case | Julia | Mean ± SD (µs) | CV | Runs |
|---|---|---:|---:|---:|
| `quadratic/dense/quadratic_model!/Float32/256/dense/warm` | julia-1.13.2 | 0.050 ± 0.075 | 150.5% | 10 |
| `quadratic/dense/quadratic_model!/Float32/32/dense/warm` | julia-1.13.2 | 0.045 ± 0.058 | 129.5% | 10 |
| `compositions/Regularize/quadratic_model!/Float64/256/dense/warm` | julia-1.13.2 | 0.043 ± 0.051 | 118.5% | 10 |
| `products/symmetric_L/Float64/256/sparse_0.01/warm` | julia-1.10.12 | 0.704 ± 0.765 | 108.6% | 10 |
| `products/symmetric_L/Float64/256/atom_0.01/warm` | julia-1.10.12 | 0.700 ± 0.751 | 107.3% | 10 |
| `coordinates/GroupNormLinf/Float32/100000/atom/warm` | julia-1.13.2 | 0.226 ± 0.216 | 95.4% | 10 |
| `products/symmetric_U/Float32/256/sparse_0.01/warm` | julia-1.10.12 | 0.436 ± 0.399 | 91.7% | 10 |
| `products/symmetric_U/Float32/256/atom_0.01/warm` | julia-1.10.12 | 0.454 ± 0.413 | 91.0% | 10 |
| `products/symmetric_U/Float64/256/sparse_0.01/warm` | julia-1.10.12 | 0.900 ± 0.817 | 90.8% | 10 |
| `products/symmetric_U/Float64/256/sparse_0.01/warm` | julia-1.13.2 | 0.976 ± 0.861 | 88.2% | 10 |
| `products/symmetric_U/Float64/256/atom_0.01/warm` | julia-1.10.12 | 0.909 ± 0.800 | 88.0% | 10 |
| `norms_sets/IndHalfspace/value_and_prox!/Float32/100000/dense/warm` | julia-1.13.2 | 23.129 ± 19.794 | 85.6% | 10 |
| `products/symmetric_U/Float64/256/atom_0.01/warm` | julia-1.13.2 | 0.976 ± 0.835 | 85.5% | 10 |
| `coordinates/GroupNormLinf/Float64/100000/atom/warm` | julia-1.13.2 | 0.237 ± 0.203 | 85.5% | 10 |
| `products/symmetric_L/Float64/256/atom_0.01/warm` | julia-1.13.2 | 0.841 ± 0.696 | 82.8% | 10 |
| `matrix_sets/IndBanded/value_and_prox!/Float64/256x256/dense/warm` | julia-1.13.2 | 14.178 ± 11.479 | 81.0% | 10 |
| `products/symmetric_L/Float64/256/sparse_0.01/warm` | julia-1.13.2 | 0.819 ± 0.659 | 80.4% | 10 |
| `coordinates/SparseGroupNormL2/Float64/100000/atom/warm` | julia-1.13.2 | 0.112 ± 0.083 | 74.2% | 10 |
| `products/symmetric_U/Float32/256/sparse_0.01/warm` | julia-1.13.2 | 0.501 ± 0.355 | 70.8% | 10 |
| `compositions/regularize/prox!/Float32/100000/dense/warm` | julia-1.13.2 | 29.934 ± 20.693 | 69.1% | 10 |
| `products/symmetric_U/Float32/256/atom_0.01/warm` | julia-1.13.2 | 0.512 ± 0.352 | 68.7% | 10 |
| `compositions/Precompose/value/Float32/32/dense/warm` | julia-1.10.12 | 0.087 ± 0.059 | 68.2% | 10 |
| `compositions/Precompose/hessian_vec!/Float32/32/dense/warm` | julia-1.13.2 | 0.063 ± 0.040 | 63.6% | 10 |
| `separable/NormL0/prox!/Float64/100000/dense/warm` | julia-1.13.2 | 33.799 ± 18.269 | 54.1% | 10 |
| `compositions/Regularize/hessian_vec!/Float32/256/dense/warm` | julia-1.13.2 | 46.665 ± 25.042 | 53.7% | 10 |

## Slowest measured calls

Absolute cost is not evidence of a defect; decompositions and initialization are expected to be expensive.

| Case | Julia | Mean (ms) | 95% interval (ms) | Runs |
|---|---|---:|---:|---:|
| `partial/trace_full/prox!/Float64/512x512/dense/warm` | julia-1.13.2 | 36.120 | [33.251, 38.989] | 10 |
| `partial/trace_full/prox!/Float64/512x512/dense/warm` | julia-1.10.12 | 34.572 | [33.730, 35.413] | 10 |
| `least_squares/tall/Float64/256/dense/construct_and_first_prox` | julia-1.13.2 | 32.998 | [29.807, 36.189] | 10 |
| `least_squares/tall/Float64/256/dense/construct_and_first_prox` | julia-1.10.12 | 30.474 | [29.620, 31.328] | 10 |
| `partial/rank4_full/prox!/Float64/512x256/dense/warm` | julia-1.13.2 | 20.757 | [19.228, 22.286] | 10 |
| `partial/nuclear_full/prox!/Float64/512x256/dense/warm` | julia-1.13.2 | 20.453 | [19.061, 21.844] | 10 |
| `partial/rank4_full/prox!/Float64/512x256/dense/warm` | julia-1.10.12 | 19.558 | [19.236, 19.880] | 10 |
| `partial/nuclear_full/prox!/Float64/512x256/dense/warm` | julia-1.10.12 | 19.461 | [19.188, 19.734] | 10 |
| `partial/trace_full/prox!/Float32/512x512/dense/warm` | julia-1.13.2 | 19.436 | [17.657, 21.216] | 10 |
| `least_squares/tall/Float32/256/dense/construct_and_first_prox` | julia-1.13.2 | 18.695 | [17.122, 20.267] | 10 |
| `partial/trace_full/prox!/Float32/512x512/dense/warm` | julia-1.10.12 | 17.765 | [17.499, 18.030] | 10 |
| `least_squares/tall/Float32/256/dense/construct_and_first_prox` | julia-1.10.12 | 17.532 | [17.033, 18.031] | 10 |
| `least_squares/wide/Float64/1024/dense/construct_and_first_prox` | julia-1.13.2 | 14.178 | [13.069, 15.287] | 10 |
| `affine_constrained/least_squares/Float64/1024/dense/construct_and_first_prox` | julia-1.10.12 | 14.098 | [13.640, 14.556] | 10 |
| `affine_constrained/least_squares/Float64/1024/dense/construct_and_first_prox` | julia-1.13.2 | 14.075 | [12.937, 15.213] | 10 |
| `least_squares/wide/Float64/1024/dense/construct_and_first_prox` | julia-1.10.12 | 13.242 | [12.722, 13.762] | 10 |
| `partial/nuclear_full/prox!/Float32/512x256/dense/warm` | julia-1.13.2 | 11.947 | [11.147, 12.747] | 10 |
| `partial/rank4_full/prox!/Float32/512x256/dense/warm` | julia-1.13.2 | 11.673 | [11.141, 12.205] | 10 |
| `partial/psd_rank4_full/prox!/Float64/512x512/dense/warm` | julia-1.13.2 | 11.560 | [10.768, 12.352] | 10 |
| `separable/KLDivergence/value_and_prox!/Float64/100000/dense/warm` | julia-1.13.2 | 11.402 | [10.983, 11.820] | 10 |
| `partial/nuclear_full/prox!/Float32/512x256/dense/warm` | julia-1.10.12 | 11.166 | [10.977, 11.354] | 10 |
| `partial/rank4_full/prox!/Float32/512x256/dense/warm` | julia-1.10.12 | 11.160 | [10.997, 11.324] | 10 |
| `separable/NegEntropy/value_and_prox!/Float64/100000/dense/warm` | julia-1.10.12 | 11.025 | [10.806, 11.245] | 10 |
| `partial/psd_rank4_full/prox!/Float64/512x512/dense/warm` | julia-1.10.12 | 10.923 | [10.534, 11.312] | 10 |
| `separable/KLDivergence/value_and_prox!/Float64/100000/dense/warm` | julia-1.10.12 | 10.686 | [10.499, 10.873] | 10 |
