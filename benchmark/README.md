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

ProximalBase at ac681eb and ProximalOperators 0.16.1 on Julia 1.13.2, one core
of an AMD EPYC 7513 (USC CARC Discovery, job 13370883, 2026-10-24). Every call
starts the search from zero, as on a new problem, and takes the steps it took
before #77; the next section times the calls of a solver, which start from the
multiplier of the last call:

| case | Constrained (μs) | passes | specialized (time, relative to Constrained) |
|:-----|---:|---:|:---|
| simplex r = 1, n = 10³                 |        2.8 |  14 | 0.7 μs (0.26×), 1.3 μs (0.46×) |
| simplex r = n/10, n = 10³              |        2.2 |  11 | 1.8 μs (0.80×), 7.3 μs (3.36×) |
| capped simplex k = 10, n = 10³         |        2.5 |  13 | 14.7 μs (5.82×) |
| capped simplex k = n/2, n = 10³        |        1.5 |   8 | 13.1 μs (8.51×) |
| zero-sum lasso λ = 0.5, n = 10³        |        2.6 |   8 |  |
| ElasticNet on the simplex, n = 10³     |        6.3 |  12 |  |
| NormL1 on [-1, 1], n = 10³             |        0.3 |   1 | 0.2 μs (0.76×) |
| simplex r = 1, n = 10⁵                 |      254.5 |  14 | 45.1 μs (0.18×), 72.8 μs (0.29×) |
| simplex r = n/10, n = 10⁵              |      203.3 |  11 | 107.7 μs (0.53×), 20329.8 μs (99.98×) |
| capped simplex k = 10, n = 10⁵         |      249.7 |  14 | 2561.4 μs (10.26×) |
| capped simplex k = n/2, n = 10⁵        |      179.9 |  10 | 2015.8 μs (11.21×) |
| zero-sum lasso λ = 0.5, n = 10⁵        |      216.6 |   7 |  |
| ElasticNet on the simplex, n = 10⁵     |      652.5 |  13 |  |
| NormL1 on [-1, 1], n = 10⁵             |       27.5 |   1 | 20.7 μs (0.75×) |

- **The capped simplex:** the search is 6 to 11 times faster than sorting.
- **The simplex:** `IndSimplex` is faster than the search in every case, by
  1.25 to 5.6 times, and remains the function for the simplex alone. Its
  threshold got up to 17 times faster when many entries stay positive (next
  section): with r = n/10 and n = 10⁵ it took 1401 μs before #77.
- **The zero-sum lasso and the elastic net on the simplex** take 7 to 13
  passes, at 0.3 to 0.5 μs per pass and 1000 entries.
- **A separable function on a box** is one pass: the clamp adds a third to the
  time of the ℓ1 norm alone.

## The search for the multiplier, and the simplex threshold

`multiplier.jl` measures the changes of issue #77:

- the search of `Constrained` on a hyperplane starts from the multiplier of the
  last call, with a first step from the exact slope of φ where the scalar
  proximal operators are piecewise linear;
- the threshold of the projection onto the simplex hands over to Michelot's
  algorithm when many entries become candidates.

The search before #77 is the current one with the slopes hidden and every call
from zero, and the threshold before #77 is a copy in the script. Each
comparison runs its sides on the same data, interleaved over rounds, and keeps
the fastest round of each side.

```sh
julia --project=benchmark benchmark/multiplier.jl
```

ProximalBase at ac681eb on Julia 1.13.2, one core of an AMD EPYC 7513 (USC CARC
Discovery, job 13370883, 2026-10-24).

**1. One call from zero**, over 200 draws of `x` at n = 10³ and 20 at n = 10⁵,
the same for both columns: the mean time and passes per call. A first call
takes the first step of the search before #77, the lower bound on the distance
to the root. The exact first step, which the search takes from the last
multiplier, is forced here from zero by storing a multiplier of zero. The last
column is the share of draws on which it takes more passes.

| case | the first step of a first call (μs, passes) | the exact first step (μs, passes) | more passes |
|:-----|---:|---:|---:|
| simplex r = 1, n = 10³ | 2.47, 12.0 | 2.38, 11.5 | 14% |
| simplex r = n/10, n = 10³ | 2.15, 10.7 | 1.86, 9.1 | 0% |
| capped simplex k = 10, n = 10³ | 2.11, 10.5 | 2.08, 10.1 | 5% |
| capped simplex k = n/2, n = 10³ | 1.59, 8.1 | 1.44, 7.0 | 4% |
| zero-sum lasso λ = 0.5, n = 10³ | 2.50, 7.4 | 1.82, 5.3 | 1% |
| ℓ1 on [-1, 1], a general normal, n = 10³ | 3.34, 7.6 | 2.73, 5.8 | 1% |
| elastic net on the simplex, n = 10³ | 5.95, 11.1 | 6.02, 10.8 | 26% |
| simplex r = 1, n = 10⁵ | 310.5, 15.9 | 308.8, 15.7 | 15% |
| simplex r = n/10, n = 10⁵ | 214.4, 11.0 | 221.0, 11.0 | 0% |
| capped simplex k = 10, n = 10⁵ | 280.1, 14.7 | 284.7, 14.4 | 10% |
| capped simplex k = n/2, n = 10⁵ | 167.6, 9.0 | 158.9, 7.8 | 5% |
| zero-sum lasso λ = 0.5, n = 10⁵ | 254.1, 7.9 | 168.0, 5.1 | 0% |
| ℓ1 on [-1, 1], a general normal, n = 10⁵ | 329.7, 7.8 | 243.3, 5.3 | 0% |
| elastic net on the simplex, n = 10⁵ | 807.4, 15.8 | 840.7, 15.9 | 40% |

- The exact first step would make a first call 18 to 34% faster on the
  zero-sum lasso and on ℓ1 with a general normal, whose root lies near zero.
- Where the root lies far out in the tail of the entries, as for the elastic
  net on the simplex, it saves under half a pass, while the pass that computes
  the slope costs half a pass more: the elastic net would be 1 to 4% slower. A
  first call therefore keeps the old first step.

**2. The calls of a solver**, on the same 20 slowly changing points for every
column: the time and passes per call. The middle column starts from the last
multiplier with the slope of the last secant, as for a function without known
slopes.

| case | before #77, each call from zero (μs, passes) | from the last multiplier, slopes hidden (μs, passes) | after #77, from the last multiplier (μs, passes) | speedup |
|:-----|---:|---:|---:|---:|
| simplex r = 1, n = 10³ | 2.10, 10.6 | 0.99, 5.2 | 0.77, 4.0 | 2.72× |
| simplex r = n/10, n = 10³ | 2.11, 10.6 | 0.86, 4.4 | 0.61, 3.2 | 3.47× |
| capped simplex k = 10, n = 10³ | 1.99, 10.2 | 0.78, 4.3 | 0.64, 3.1 | 3.09× |
| capped simplex k = n/2, n = 10³ | 1.51, 7.9 | 0.78, 4.4 | 0.64, 3.1 | 2.38× |
| zero-sum lasso λ = 0.5, n = 10³ | 2.58, 7.9 | 1.35, 4.5 | 1.03, 3.1 | 2.50× |
| ℓ1 on [-1, 1], a general normal, n = 10³ | 3.42, 7.8 | 2.06, 4.8 | 1.47, 3.1 | 2.33× |
| elastic net on the simplex, n = 10³ | 5.93, 11.2 | 2.53, 4.9 | 2.12, 3.7 | 2.80× |
| simplex r = 1, n = 10⁵ | 306.4, 15.3 | 106.6, 5.1 | 87.0, 4.2 | 3.52× |
| simplex r = n/10, n = 10⁵ | 221.2, 11.0 | 101.2, 5.0 | 91.7, 4.5 | 2.41× |
| capped simplex k = 10, n = 10⁵ | 282.8, 14.9 | 128.7, 5.7 | 90.0, 4.0 | 3.14× |
| capped simplex k = n/2, n = 10⁵ | 178.2, 9.3 | 98.5, 4.9 | 87.3, 4.1 | 2.04× |
| zero-sum lasso λ = 0.5, n = 10⁵ | 246.1, 7.5 | 151.5, 4.8 | 125.2, 3.8 | 1.97× |
| ℓ1 on [-1, 1], a general normal, n = 10⁵ | 334.8, 7.8 | 211.8, 4.8 | 180.4, 3.9 | 1.86× |
| elastic net on the simplex, n = 10⁵ | 799.7, 15.6 | 251.8, 5.0 | 227.8, 4.1 | 3.51× |

- From the last multiplier, a call takes 3.1 to 4.5 passes instead of 7.5 to
  15.6, and is 1.9 to 3.5 times faster.
- The exact slope saves 0.5 to 1.7 passes per call against the slope of the
  last secant.

**3. FISTA on the zero-sum lasso**: the points at which 500 iterations call the
proximal operator, recorded from one run and replayed for both sides.

| problem | an iteration before #77 (ms) | before #77, each call from zero (μs, passes) | after #77, from the last multiplier (μs, passes) | speedup |
|:-----|---:|---:|---:|---:|
| n = 100, p = 20000 | 1.00 | 61.3, 9.1 | 22.8, 3.1 | 2.69× |
| n = 50, p = 100000 | 4.80 | 289.1, 8.5 | 123.6, 3.4 | 2.34× |

- The calls take 3.1 to 3.4 passes instead of 8.5 to 9.1, and are 2.3 to 2.7
  times faster. Before #77 they took 6% of an iteration; the products with the
  data matrix take the rest.

**4. The breakpoint search** that #77 proposed (Brucker, 1984; Kiwiel, 2008)
selects the median of the breakpoints left in the bracket, several times. One
selection over the 2n breakpoints of the zero-sum lasso, with the in-place
selection of `IndBallL0`, against the whole search from zero:

| n | fill 2n breakpoints and select their median (μs) | the whole search from zero (μs) |
|---:|---:|---:|
| 1000 | 3.0 | 2.3 |
| 100000 | 1019.0 | 254.8 |

- One selection costs 1.3 to 4 times as much as the whole search, so the
  breakpoint search was not implemented.

**5. The threshold of the projection onto the simplex**, Condat's pass before
#77 against the current one, on the same inputs:

| case | before #77 (μs) | after #77 (μs) | speedup | difference in τ |
|:-----|---:|---:|---:|---:|
| r = 1, n = 10³ | 0.6 | 0.5 | 1.22× | 1e-15 |
| r = 100, n = 10³ | 6.0 | 1.6 | 3.67× | 2e-15 |
| r = n/100, n = 10³ | 2.3 | 0.8 | 2.99× | 8e-15 |
| r = n/10, n = 10³ | 6.0 | 1.7 | 3.61× | 2e-15 |
| r = n/2, n = 10³ | 9.0 | 2.6 | 3.53× | 5e-16 |
| r = 1, n = 10⁵ | 39.1 | 28.8 | 1.36× | 9e-16 |
| r = 100, n = 10⁵ | 187.6 | 135.1 | 1.39× | 1e-13 |
| r = n/100, n = 10⁵ | 562.0 | 300.4 | 1.87× | 2e-14 |
| r = n/10, n = 10⁵ | 1286.5 | 105.5 | 12.20× | 7e-15 |
| r = n/2, n = 10⁵ | 1384.2 | 83.1 | 16.66× | 2e-14 |

- 1.2 to 17 times faster, most when many entries stay positive, with the same
  threshold up to 1e-13. `IndSimplex`, `IndBallL1`, `NormLinf` and
  `GroupNormLinf` use it.

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
