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

## Constrained

`constrained.jl` times the proximal operators of `Constrained` against the
specialized algorithms for the same sets: the projections onto the simplex of
ProximalBase (Condat's algorithm) and of ProximalOperators, and a projection
onto the capped simplex that sorts the breakpoints, in O(n log n). The passes
are those of the search for the multiplier over `x`, counted as calls of the
scalar proximal operator of `f`.

```sh
julia --project=benchmark benchmark/constrained.jl
```

ProximalBase at c8c083c and ProximalOperators 0.16.1 on Julia 1.13.2, one core
of an AMD EPYC 7513 (USC CARC Discovery, job 13350846, 2026-10-23):

| case | Constrained (μs) | passes | specialized (time, relative to Constrained) |
|:-----|---:|---:|:---|
| simplex r = 1, n = 10³                 |        2.9 |  14 | 1.0 μs (0.36×), 1.4 μs (0.47×) |
| simplex r = n/10, n = 10³              |        2.3 |  11 | 6.4 μs (2.79×), 8.0 μs (3.51×) |
| capped simplex k = 10, n = 10³         |        2.7 |  13 | 16.0 μs (5.89×) |
| capped simplex k = n/2, n = 10³        |        1.6 |   8 | 14.3 μs (8.73×) |
| zero-sum lasso λ = 0.5, n = 10³        |        2.9 |   8 |  |
| ElasticNet on the simplex, n = 10³     |        6.6 |  12 |  |
| NormL1 on [-1, 1], n = 10³             |        0.3 |   1 | 0.2 μs (0.75×) |
| simplex r = 1, n = 10⁵                 |      274.1 |  14 | 62.2 μs (0.23×), 79.2 μs (0.29×) |
| simplex r = n/10, n = 10⁵              |      215.8 |  11 | 1400.9 μs (6.49×), 21423.4 μs (99.26×) |
| capped simplex k = 10, n = 10⁵         |      267.8 |  14 | 2767.8 μs (10.33×) |
| capped simplex k = n/2, n = 10⁵        |      189.5 |  10 | 2208.2 μs (11.65×) |
| zero-sum lasso λ = 0.5, n = 10⁵        |      241.3 |   7 |  |
| ElasticNet on the simplex, n = 10⁵     |      673.9 |  13 |  |
| NormL1 on [-1, 1], n = 10⁵             |       29.9 |   1 | 22.3 μs (0.75×) |

- **The capped simplex:** the search is 6 to 12 times faster than sorting.
- **The simplex:** when most entries stay positive (r = n/10), the search is
  2.8 to 6.5 times faster than Condat's algorithm, whose list of candidate
  entries is then long. When few entries stay positive (r = 1), Condat's
  algorithm is 2.8 to 4.4 times faster, and `IndSimplex` remains the function
  for the simplex alone.
- **The zero-sum lasso and the elastic net on the simplex** take 7 to 13
  passes, at 0.35 to 0.5 μs per pass and 1000 entries.
- **A separable function on a box** is one pass: the clamp adds a third to the
  time of the ℓ1 norm alone.

## Spectral

`spectral.jl` times `Spectral(f)` against the functions of the eigenvalues
that ProximalBase and ProximalOperators have, and the projection onto the
Fantope against the same computation written with LinearAlgebra's `eigen`.

```sh
julia --project=benchmark benchmark/spectral.jl
```

ProximalBase at dac7e5d and ProximalOperators 0.16.1 on Julia 1.13.2, one core
of an AMD EPYC 7513 with one BLAS thread (USC CARC Discovery, job 13351030,
2026-10-23):

| case | Spectral (μs) | bytes | others (time, relative to Spectral; bytes) |
|:-----|---:|---:|:---|
| PSD cone, 50 × 50              |      197 |       0 | 196 μs (1.00×), 0 bytes; 239 μs (1.21×), 59136 bytes |
| trace norm, 50 × 50            |      199 |       0 | 328 μs (1.65×), 0 bytes; 327 μs (1.65×), 148224 bytes |
| trace on the cone, 50 × 50     |      197 |       0 | 196 μs (0.99×), 0 bytes |
| Fantope k = 5, 50 × 50         |      194 |       0 | 204 μs (1.05×), 79736 bytes |
| eigenvalues in [-1, 1], 50 × 50 |      199 |       0 |  |
| PSD cone, 200 × 200            |     3099 |       0 | 3112 μs (1.00×), 0 bytes; 6612 μs (2.13×), 714216 bytes |
| trace norm, 200 × 200          |     3192 |       0 | 5510 μs (1.73×), 0 bytes; 5537 μs (1.73×), 2266592 bytes |
| trace on the cone, 200 × 200   |     3100 |       0 | 3106 μs (1.00×), 0 bytes |
| Fantope k = 5, 200 × 200       |     3017 |       0 | 3493 μs (1.16×), 1035872 bytes |
| eigenvalues in [-1, 1], 200 × 200 |     3188 |       0 |  |

- `Spectral` allocates nothing in any case, and costs what the eigendecomposition
  costs: the same as `IndPSD` and `TracePSD`.
- On a symmetric matrix, the trace norm by the eigendecomposition is 1.7 times
  faster than `NuclearNorm`, which computes an SVD.
- The projection onto the Fantope is 1.05 to 1.16 times faster than the one
  written with `eigen`, which allocates 80 kB at n = 50 and 1 MB at n = 200 on
  every call.
