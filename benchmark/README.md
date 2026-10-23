# Benchmarks

`compare.jl` times the proximal operators of ProximalBase against
[ProximalOperators.jl](https://github.com/JuliaFirstOrder/ProximalOperators.jl),
the general Julia library of proximal operators, and counts the bytes each call
allocates. Every case calls `prox!(y, f, x, γ)` on a fixed input after a
warm-up, as a solver does; the times are medians from BenchmarkTools.

Run from the repository root:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/compare.jl
```

## Results

ProximalBase 0.5.0 and ProximalOperators 0.16.1 on Julia 1.13.2, Apple M3 Pro
(2026-10-23, after the review fixes):

| case | ProximalBase (μs) | ProximalOperators (μs) | speedup | PB bytes | PO bytes |
|:-----|---:|---:|---:|---:|---:|
| NormL1, n = 10⁵                    |       17.4 |       34.1 |  1.96× |         0 |         0 |
| weighted NormL1, n = 10⁵           |       20.5 |       57.6 |  2.80× |         0 |    802880 |
| NormL2, n = 10⁵                    |       20.0 |      116.2 |  5.81× |         0 |         0 |
| ElasticNet, n = 10⁵                |       26.4 |       77.0 |  2.92× |         0 |         0 |
| NormL0, n = 10⁵                    |       10.5 |      123.2 | 11.69× |         0 |         0 |
| NormLinf, n = 10⁵                  |       55.7 |      138.5 |  2.49× |         0 |   1607648 |
| IndBallL1(10), n = 10⁵             |       84.4 |      146.0 |  1.73× |         0 |    853568 |
| IndSimplex(1), n = 10⁵             |       38.7 |       54.8 |  1.42× |         0 |      5552 |
| IndBallL0(1000), n = 10⁵           |      102.1 |      290.2 |  2.84× |         0 |   1605808 |
| IndBox(-1, 1), n = 10⁵             |       20.8 |       81.0 |  3.90× |         0 |         0 |
| group ℓ2, 10⁴ groups of 10         |       35.5 |       45.0 |  1.27× |         0 |         0 |
| NuclearNorm, 200 × 100             |     1422.8 |     1422.7 |  1.00× |         0 |    829728 |
| IndBallRank(5), 200 × 100          |     1408.4 |     2186.6 |  1.55× |         0 |   2443200 |
| IndPSD, 200 × 200                  |     1873.8 |     5276.5 |  2.82× |         0 |    742944 |
| LeastSquares prox, 500 × 200       |       10.5 |       57.3 |  5.46× |         0 |         0 |
| LeastSquares prox, γ changing      |       10.4 |      117.3 | 11.31× |         0 |    655520 |

Times vary by about 10% between runs on a machine with other work; the time
of ProximalOperators' least-squares proximal operator with a fixed step varied
between 23 and 57 μs, against 10 μs for ProximalBase in every run.

ProximalBase allocates nothing in any case. The main sources of its speed:

- separable penalties are one fused pass over the entries;
- the projections onto the simplex and the ℓ1 ball, and the proximal operator
  of the ℓ∞ norm, use Condat's algorithm with a scratch buffer kept by the
  function, in a loop over an index range (a loop through `iterate` is three
  times slower);
- `IndBallL0` selects the `s` largest entries with Floyd and Rivest's
  algorithm, in place, and keeps them in one vectorized pass (ties beyond `s`
  take a second pass only when there are some);
- the spectral functions call LAPACK with workspaces kept by the function, and
  `IndPSD` rebuilds the projection with `syrk` from the eigenvectors that remain;
- the proximal operator of `LeastSquaresLoss` uses a thin SVD of `X` computed
  once, so a step that changes, as in ADMM with an adaptive penalty, costs
  nothing more; ProximalOperators refactors its Cholesky factor at each new step.

`NuclearNorm` is bound by the SVD in both packages.

## Against ProximalBase 0.4

`compare_04.jl` times the calls that the dependents of ProximalBase make, in
whichever version its environment has: with ProximalBase 0.4.0
(`Pkg.add(name = "ProximalBase", version = "0.4.0")`) or with this checkout.
In 0.4 a solver called `prox!` and then the value, which for the spectral
functions meant a second decomposition; 0.5 has `value_and_prox!`.

| call | 0.4.0 | 0.5.0 |
|:-----|------:|------:|
| ℓ1 prox, n = 10⁵ | 20.0 μs | 16.7 μs |
| group ℓ2 prox, 10⁴ groups of 10 | 114.9 μs | 35.5 μs |
| nuclear norm, prox and value, 200 × 100 | 2119 μs, 984 KB | 1415 μs, 0 bytes |
| Gaussian likelihood, prox and value, 200 × 200 | 2099 μs, 995 KB | 1914 μs, 0 bytes |
| least squares, value and gradient, 500 × 200 | 88.2 μs | 26.6 μs |
| ℓ1 prox of a SparseIterate with 1% nonzeros, n = 10⁵ | 89.6 μs | 0.8 μs |

The least-squares loss gains from a buffer of a concrete type (0.4 stored a
`Union` of vector and matrix), and the sparse iterate from proximal operators
that touch only the stored entries.

## Functions of the order of the entries

[`ordered/run.jl`](ordered/run.jl) compares `TotalVariation1D` with
ProximalOperators' implementation of the same algorithm, `IndIsotonic` with
NearlyIsotonicRegression.jl's pool adjacent violators, and `SLOPE` with
SLOPE.jl, the interface to libslope. SLOPE.jl solves the regression, so FISTA
with the proximal operator of `SLOPE` solves the same problem,
`‖y - Xβ‖²/(2n) + α Σᵢ λᵢ|β|₍ᵢ₎`. The packages it needs have their own
environment:

```sh
julia --project=benchmark/ordered -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark/ordered benchmark/ordered/run.jl
```

ProximalBase at 9cf0c03, ProximalOperators 0.16.1, NearlyIsotonicRegression
0.1.0 and SLOPE 1.3.1 on Julia 1.13.2, one core of an AMD EPYC 7513 with one
BLAS thread (USC CARC Discovery, job 13351505, 2026-10-23):

| case | ProximalBase (μs) | other | other (μs) | ratio | largest difference |
|:-----|---:|:---|---:|---:|---:|
| TV, noise, n = 10³                 |       5.3 | ProximalOperators          |       6.2 |  1.17× | 2e-16 |
| TV, 10 steps and noise, n = 10³    |       6.7 | ProximalOperators          |       7.5 |  1.12× | 2e-16 |
| isotonic, n = 10³                  |       2.9 | NearlyIsotonicRegression   |      10.1 |  3.51× | 2e-15 |
| SLOPE, n = 10³                     |      13.0 | NormL1 of ProximalBase     |       0.2 |  0.02× | NaN |
| TV, noise, n = 10⁵                 |    1848.5 | ProximalOperators          |    1804.8 |  0.98× | 4e-16 |
| TV, 10 steps and noise, n = 10⁵    |    1679.0 | ProximalOperators          |    1640.6 |  0.98× | 4e-16 |
| isotonic, n = 10⁵                  |    1414.9 | NearlyIsotonicRegression   |    3295.9 |  2.33× | 1e-12 |
| SLOPE, n = 10⁵                     |    6464.7 | NormL1 of ProximalBase     |      20.6 |  0.00× | NaN |

| SLOPE regression | FISTA with SLOPE (ms) | iterations | SLOPE.jl (ms) | ratio | largest difference, 1/n loss | 1/1 loss |
|:-----|---:|---:|---:|---:|---:|---:|
| n = 200, p = 500 | 8.9 | 212 | 0.6 | 0.07× | 3e-08 | 1e+00 |
| n = 1000, p = 2000 | 101.8 | 146 | 6.9 | 0.07× | 3e-08 | 2e+00 |

- **Total variation:** as fast as ProximalOperators, with the same results to
  rounding. Condat's algorithm goes back over a segment when it ends, and at
  n = 10⁵ both implementations take about 17 ns per entry.
- **Isotonic regression:** 2.3 to 3.5 times faster than
  NearlyIsotonicRegression.jl, with the same results.
- **The proximal operator of SLOPE:** most of its time is the sort. It sorts an
  index vector by a quicksort up to 4096 entries, and with Base's `sortperm!`
  and a scratch buffer beyond, where that is 2 to 2.6 times faster; neither
  allocates.
- **SLOPE regression:** FISTA with the proximal operator of `SLOPE` finds
  SLOPE.jl's solution to 3e-8, with the loss divided by 2n. libslope's
  coordinate descent is 14 times faster than FISTA: a coordinate method for
  SLOPE belongs in the solvers.
