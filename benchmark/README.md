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

## Partial decompositions

`partial.jl` times `NuclearNorm`, `IndBallRank`, `IndPSD` and `TracePSD` with
`partial = true` against their full decompositions, and against the partial
SVDs and eigensolvers of registered packages. The input is a rank-`k` signal,
with singular values or eigenvalues from 10 to 20, plus noise of spectral norm
about 0.2; the threshold is 1. Run from the repository root:

```sh
julia --project=benchmark/partial -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
OPENBLAS_NUM_THREADS=1 julia --project=benchmark/partial benchmark/partial.jl [p ...]
```

- *full*: the same function with `partial = false`;
- *partial, cold*: a call from a random block, with the size of the block
  learned in earlier calls;
- *partial, warm*: a call on an input changed by about 0.002 in spectral norm
  since the previous call, as in a solver;
- the packages are told the exact rank and start from scratch at every call;
  their times include forming the result from their pairs.

Each time is the fastest of one to five calls. Julia 1.13.2, one thread of an
AMD EPYC 7513 (2026-10-24, commit 3e6f5a8); at `p = 5000` the full SVD and
eigendecomposition were timed once, at `k = 10`:

**`NuclearNorm` (`IndBallRank` alike): milliseconds per call**

| p | k | full | partial, cold | partial, warm | ARPACK `svds` | KrylovKit `svdsolve` | TSVD `tsvd` | LowRankApprox `psvd` (error) |
|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| 200 | 2 | 5.21 | 0.88 | 0.55 | 0.29 | 0.41 | 0.13 | 0.28 (3.6e-02) |
| 200 | 10 | 5.20 | 1.37 | 0.84 | 0.40 | 0.43 | 0.34 | 0.42 (3.3e-02) |
| 200 | 50 | 5.00 | 5.04 | 5.07 | 3.04 | — | 1.75 | 2.05 (4.0e-02) |
| 500 | 2 | 49.8 | 3.89 | 2.46 | 2.07 | 2.60 | 1.00 | 1.42 (5.0e-02) |
| 500 | 10 | 49.7 | 5.42 | 3.39 | 2.40 | 2.71 | 1.06 | 1.14 (5.2e-02) |
| 500 | 50 | 48.5 | 27.5 | 16.8 | 9.43 | — | 4.90 | 4.89 (7.6e-02) |
| 1000 | 2 | 313 | 14.3 | 8.90 | 5.84 | 7.51 | 1.62 | 3.06 (6.4e-02) |
| 1000 | 10 | 313 | 18.8 | 11.7 | 7.00 | 8.00 | 7.11 | 4.34 (7.0e-02) |
| 1000 | 50 | 311 | 72.2 | 44.5 | 31.9 | — | 17.1 | 13.5 (1.0e-01) |
| 2000 | 2 | 2,338 | 60.6 | 49.1 | 31.9 | 50.9 | 8.86 | 12.8 (9.4e-02) |
| 2000 | 10 | 2,336 | 74.9 | 46.2 | 33.9 | 41.3 | 20.0 | 14.2 (1.0e-01) |
| 2000 | 50 | 2,302 | 231 | 142 | 155 | — | 85.3 | 43.6 (1.3e-01) |
| 5000 | 2 | 43,042 | 403 | 322 | 344 | 454 | 126 | 134 (1.4e-01) |
| 5000 | 10 | 43,042 | 487 | 343 | 387 | 681 | 299 | 151 (1.5e-01) |
| 5000 | 50 | 43,042 | 1,235 | 780 | 1,548 | — | 907 | 295 (2.2e-01) |

**`IndPSD(rank = k)`: milliseconds per call**

| p | k | full (`syevr`) | partial, cold | partial, warm | ARPACK `eigs` | KrylovKit `eigsolve` |
|--:|--:|--:|--:|--:|--:|--:|
| 200 | 2 | 1.02 | 0.57 | 0.37 | 0.34 | 0.34 |
| 200 | 10 | 1.62 | 1.03 | 0.65 | 0.37 | 0.37 |
| 200 | 50 | 4.42 | 4.40 | 4.40 | 2.35 | — |
| 500 | 2 | 8.76 | 2.44 | 1.60 | 0.67 | 0.90 |
| 500 | 10 | 10.3 | 3.72 | 2.42 | 0.99 | 1.03 |
| 500 | 50 | 17.6 | 15.4 | 9.84 | 5.12 | — |
| 1000 | 2 | 52.1 | 8.70 | 6.80 | 4.74 | 5.50 |
| 1000 | 10 | 54.9 | 12.4 | 9.63 | 3.45 | 3.46 |
| 1000 | 50 | 70.8 | 41.2 | 31.4 | 14.6 | — |
| 2000 | 2 | 358 | 36.0 | 26.5 | 10.0 | 22.6 |
| 2000 | 10 | 364 | 48.1 | 31.8 | 13.6 | 14.2 |
| 2000 | 50 | 404 | 134 | 86.6 | 56.8 | — |
| 5000 | 2 | 7,098 | 370 | 267 | 138 | 174 |
| 5000 | 10 | 7,365 | 430 | 303 | 174 | 202 |
| 5000 | 50 | 7,694 | 912 | 646 | 479 | — |

**`TracePSD` (`IndPSD()` alike): milliseconds per call**

| p | k | full (`syevd`) | partial, cold | partial, warm |
|--:|--:|--:|--:|--:|
| 200 | 2 | 2.99 | 0.82 | 0.53 |
| 200 | 10 | 2.92 | 1.34 | 0.84 |
| 200 | 50 | 2.83 | 2.82 | 2.81 |
| 500 | 2 | 24.9 | 3.43 | 2.22 |
| 500 | 10 | 25.1 | 4.87 | 3.14 |
| 500 | 50 | 24.5 | 26.1 | 16.6 |
| 1000 | 2 | 148 | 12.2 | 7.94 |
| 1000 | 10 | 148 | 16.1 | 10.5 |
| 1000 | 50 | 146 | 64.8 | 40.7 |
| 2000 | 2 | 1,006 | 48.8 | 32.0 |
| 2000 | 10 | 1,012 | 62.0 | 40.2 |
| 2000 | 50 | 1,013 | 195 | 125 |
| 5000 | 2 | 16,722 | 401 | 314 |
| 5000 | 10 | 16,722 | 538 | 311 |
| 5000 | 50 | 16,722 | 1,079 | 699 |

`IndPSD()` is tested on the same matrices shifted by `-0.5I`, so that it keeps
the `k` eigenvalues above 0.5; its times are within 20% of those of
`TracePSD`. No partial call allocates. The largest relative error of a partial
result against the full decomposition is 2.2e-12 (the warm results at
`p = 5000` are not checked). At `p = 200` with `k = 50` the block exceeds a
quarter of the dimension, and every function takes the full decomposition.

The packages:

- ARPACK and KrylovKit are accurate to 1e-15. KrylovKit, with its default
  Krylov dimension of 30, refuses `k = 50`.
- TSVD's errors are 3e-11 to 6e-7 at its default tolerance.
- The randomized decompositions without power iterations are fast and
  inaccurate: errors from 2e-2 to 2e-1 for LowRankApprox's `psvd` and
  `pheigfact`, and from 2e-2 to 1.4e-1 for RandomizedLinAlg's `rsvd`.
- All of them allocate at every call: about their Krylov basis or sketch,
  besides the result.

The partial path is therefore implemented in ProximalBase rather than taken
from a package: the packages need the rank in advance, which `NuclearNorm`,
`TracePSD` and `IndPSD()` do not know; they allocate at every call; they start
from one vector, not from the block of the previous call; and only ARPACK and
KrylovKit agree with the full path. On one matrix of known rank ARPACK is
faster: its `eigs` by 1.1 to 2.8 times than a warm partial `IndPSD(rank = k)`,
and its `svds` by up to 2 times than the partial SVD for `p ≤ 2000`, though it
is up to 2 times slower at `p = 5000`.
