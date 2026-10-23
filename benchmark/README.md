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

## The nearest correlation matrix (`intersection.jl`)

`intersection.jl` projects an indefinite symmetric matrix with unit diagonal
onto the correlation matrices with `IndIntersection(IndPSD(), IndBox(L, U))`
(Dykstra's algorithm), from zero corrections ("cold") and from the corrections
of a call on an input `2e-4` away ("warm", as inside a solver). It compares
them with the quadratically convergent Newton method of Qi and Sun (2006) and
with alternating projections, both from NearestCorrelationMatrix.jl:

```sh
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
julia --project=benchmark benchmark/intersection.jl
```

ProximalBase at 62067d9, NearestCorrelationMatrix 1.2.1 and Julia 1.13.2 on
one core of an AMD EPYC 7513 (USC Discovery), one BLAS thread (2026-10-23):

| p | Dykstra, cold (ms) | rounds | Dykstra, warm (ms) | rounds | Newton (ms) | alternating projections (ms) | ‖Dykstra − Newton‖ |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | 4.02 | 19 | 2.67 | 13 | 1.19 | 3.35 | 5.6e-08 |
| 200 | 59.86 | 17 | 39.51 | 11 | 26.54 | 57.68 | 6.9e-10 |
| 500 | 553.54 | 17 | 342.18 | 11 | 253.52 | 589.69 | 4.7e-07 |

Each round of Dykstra's algorithm costs one eigendecomposition, as an
iteration of alternating projections does, and the two take the same time.
Newton's method is 2.2 to 3.4 times faster from scratch, as expected for this
particular problem, and 1.3 to 2.2 times faster than a warm start. So for
repeated correlation repair alone, that package is the better tool.
`IndIntersection` is for intersections that have no dedicated solver, and its
warm start pays inside a solver whose inputs converge.

