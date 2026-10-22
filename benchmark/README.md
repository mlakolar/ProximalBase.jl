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
(2026-10-23):

| case | ProximalBase (μs) | ProximalOperators (μs) | speedup | PB bytes | PO bytes |
|:-----|---:|---:|---:|---:|---:|
| NormL1, n = 10⁵                    |       17.0 |       34.1 |  2.01× |         0 |         0 |
| weighted NormL1, n = 10⁵           |       21.2 |       57.9 |  2.74× |         0 |    802880 |
| NormL2, n = 10⁵                    |       22.8 |      120.4 |  5.28× |         0 |         0 |
| ElasticNet, n = 10⁵                |       27.5 |       80.4 |  2.92× |         0 |         0 |
| NormL0, n = 10⁵                    |       11.4 |      137.9 | 12.12× |         0 |         0 |
| NormLinf, n = 10⁵                  |       56.2 |      138.4 |  2.46× |         0 |   1607648 |
| IndBallL1(10), n = 10⁵             |       84.5 |      145.7 |  1.72× |         0 |    853568 |
| IndSimplex(1), n = 10⁵             |       42.2 |       54.0 |  1.28× |         0 |      5552 |
| IndBallL0(1000), n = 10⁵           |      141.6 |      306.9 |  2.17× |         0 |   1605808 |
| IndBox(-1, 1), n = 10⁵             |       21.2 |       66.7 |  3.14× |         0 |         0 |
| group ℓ2, 10⁴ groups of 10         |       37.6 |       46.0 |  1.22× |         0 |         0 |
| NuclearNorm, 200 × 100             |     1421.3 |     1424.2 |  1.00× |         0 |    829728 |
| IndBallRank(5), 200 × 100          |     1405.1 |     2192.0 |  1.56× |         0 |   2328720 |
| IndPSD, 200 × 200                  |     1879.3 |     5198.1 |  2.77× |         0 |    742944 |
| LeastSquares prox, 500 × 200       |       10.5 |       56.9 |  5.42× |         0 |         0 |
| LeastSquares prox, γ changing      |       10.5 |      111.5 | 10.62× |         0 |    655520 |

ProximalBase allocates nothing in any case. The main sources of its speed:

- separable penalties are one fused pass over the entries;
- the projections onto the simplex and the ℓ1 ball, and the proximal operator
  of the ℓ∞ norm, use Condat's algorithm with a scratch buffer kept by the
  function, in a loop over an index range (a loop through `iterate` is three
  times slower);
- `IndBallL0` selects the `s` largest entries with Floyd and Rivest's
  algorithm, in place;
- the spectral functions call LAPACK with workspaces kept by the function, and
  `IndPSD` rebuilds the projection with `syrk` from the eigenvectors that remain;
- the proximal operator of `LeastSquaresLoss` uses a thin SVD of `X` computed
  once, so a step that changes, as in ADMM with an adaptive penalty, costs
  nothing more; ProximalOperators refactors its Cholesky factor at each new step.

`NuclearNorm` is bound by the SVD in both packages.
