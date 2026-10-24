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

ProximalBase at 5506b9b and ProximalOperators 0.16.1 on Julia 1.13.2, one core
of an AMD EPYC 7513 (USC CARC Discovery, job 13371989, 2026-10-24). Every call
starts the search from zero, as on a new problem, and takes the steps it took
before #77; the next section times the calls of a solver, which start from the
multiplier of the last call:

| case | Constrained (μs) | passes | specialized (time, relative to Constrained) |
|:-----|---:|---:|:---|
| simplex r = 1, n = 10³                 |        2.8 |  14 | 0.4 μs (0.15×), 1.3 μs (0.46×) |
| simplex r = n/10, n = 10³              |        2.2 |  11 | 1.8 μs (0.84×), 7.5 μs (3.41×) |
| capped simplex k = 10, n = 10³         |        2.5 |  13 | 15.2 μs (6.07×) |
| capped simplex k = n/2, n = 10³        |        1.5 |   8 | 13.4 μs (8.77×) |
| zero-sum lasso λ = 0.5, n = 10³        |        2.6 |   8 |  |
| ElasticNet on the simplex, n = 10³     |        6.3 |  12 |  |
| NormL1 on [-1, 1], n = 10³             |        0.3 |   1 | 0.2 μs (0.76×) |
| simplex r = 1, n = 10⁵                 |      256.4 |  14 | 32.2 μs (0.13×), 73.9 μs (0.29×) |
| simplex r = n/10, n = 10⁵              |      203.8 |  11 | 110.0 μs (0.54×), 20381.2 μs (99.98×) |
| capped simplex k = 10, n = 10⁵         |      250.8 |  14 | 2694.8 μs (10.74×) |
| capped simplex k = n/2, n = 10⁵        |      181.0 |  10 | 2112.9 μs (11.67×) |
| zero-sum lasso λ = 0.5, n = 10⁵        |      216.5 |   7 |  |
| ElasticNet on the simplex, n = 10⁵     |      653.5 |  13 |  |
| NormL1 on [-1, 1], n = 10⁵             |       27.5 |   1 | 20.7 μs (0.75×) |

- **The capped simplex:** the search is 6 to 12 times faster than sorting.
- **The simplex:** `IndSimplex` is faster than the search in every case, by
  1.2 to 7.7 times, and remains the function for the simplex alone. Its
  threshold got up to 16 times faster (next section): with r = n/10 and
  n = 10⁵ it took 1401 μs before #77, and takes 110 μs.
- **The zero-sum lasso and the elastic net on the simplex** take 7 to 13
  passes, at 0.3 to 0.5 μs per pass and 1000 entries.
- **A separable function on a box** is one pass: the clamp adds a third to the
  time of the ℓ1 norm alone.

## The search for the multiplier, and the simplex threshold

`multiplier.jl` measures the changes of issue #77:

- the search of `Constrained` on a hyperplane starts from the multiplier of the
  last call, with a first step from the exact slope of φ where the scalar
  proximal operators are piecewise linear;
- the threshold of the projection onto the simplex tests four entries at a
  time, and hands over to Michelot's algorithm when many entries become
  candidates.

The search before #77 is the current one with the slopes hidden and every call
from zero, and the threshold before #77 is a copy in the script. Each
comparison runs its sides on the same data, interleaved over rounds, and keeps
the fastest round of each side.

```sh
julia --project=benchmark benchmark/multiplier.jl
```

ProximalBase at 5506b9b on Julia 1.13.2, one core of an AMD EPYC 7513 (USC CARC
Discovery, job 13371989, 2026-10-24).

**1. One call from zero**, over 200 draws of `x` at n = 10³ and 20 at n = 10⁵,
the same for both columns: the mean time and passes per call. A first call
takes the first step of the search before #77, the lower bound on the distance
to the root. The exact first step, which the search takes from the last
multiplier, is forced here from zero by storing a multiplier of zero. The last
column is the share of draws on which it takes more passes.

| case | the first step of a first call (μs, passes) | the exact first step (μs, passes) | more passes |
|:-----|---:|---:|---:|
| simplex r = 1, n = 10³ | 2.39, 12.0 | 2.33, 11.5 | 14% |
| simplex r = n/10, n = 10³ | 2.13, 10.7 | 1.88, 9.1 | 0% |
| capped simplex k = 10, n = 10³ | 2.00, 10.5 | 2.03, 10.1 | 5% |
| capped simplex k = n/2, n = 10³ | 1.58, 8.1 | 1.44, 7.0 | 4% |
| zero-sum lasso λ = 0.5, n = 10³ | 2.54, 7.4 | 1.86, 5.3 | 1% |
| ℓ1 on [-1, 1], a general normal, n = 10³ | 3.33, 7.6 | 2.71, 5.8 | 1% |
| elastic net on the simplex, n = 10³ | 5.89, 11.1 | 5.97, 10.8 | 26% |
| simplex r = 1, n = 10⁵ | 340.9, 15.9 | 337.3, 15.7 | 15% |
| simplex r = n/10, n = 10⁵ | 215.8, 11.0 | 220.9, 11.0 | 0% |
| capped simplex k = 10, n = 10⁵ | 307.8, 14.7 | 307.2, 14.4 | 10% |
| capped simplex k = n/2, n = 10⁵ | 173.7, 9.0 | 162.4, 7.8 | 5% |
| zero-sum lasso λ = 0.5, n = 10⁵ | 249.3, 7.9 | 165.0, 5.1 | 0% |
| ℓ1 on [-1, 1], a general normal, n = 10⁵ | 346.5, 7.8 | 254.4, 5.3 | 0% |
| elastic net on the simplex, n = 10⁵ | 806.1, 15.8 | 840.8, 15.9 | 40% |

- The exact first step would make a first call 19 to 34% faster on the
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
| simplex r = 1, n = 10³ | 2.10, 10.6 | 0.98, 5.2 | 0.77, 4.0 | 2.73× |
| simplex r = n/10, n = 10³ | 2.11, 10.6 | 0.86, 4.4 | 0.60, 3.2 | 3.50× |
| capped simplex k = 10, n = 10³ | 1.99, 10.2 | 0.77, 4.3 | 0.63, 3.1 | 3.13× |
| capped simplex k = n/2, n = 10³ | 1.51, 7.9 | 0.77, 4.4 | 0.63, 3.1 | 2.41× |
| zero-sum lasso λ = 0.5, n = 10³ | 2.58, 7.9 | 1.34, 4.5 | 1.02, 3.1 | 2.54× |
| ℓ1 on [-1, 1], a general normal, n = 10³ | 3.47, 7.8 | 2.09, 4.8 | 1.49, 3.1 | 2.33× |
| elastic net on the simplex, n = 10³ | 5.93, 11.2 | 2.53, 4.9 | 2.12, 3.7 | 2.80× |
| simplex r = 1, n = 10⁵ | 295.7, 15.3 | 103.8, 5.1 | 90.0, 4.2 | 3.28× |
| simplex r = n/10, n = 10⁵ | 216.6, 11.0 | 101.2, 5.0 | 94.5, 4.5 | 2.29× |
| capped simplex k = 10, n = 10⁵ | 283.2, 14.9 | 111.7, 5.7 | 90.3, 4.0 | 3.14× |
| capped simplex k = n/2, n = 10⁵ | 178.0, 9.3 | 96.5, 4.9 | 90.7, 4.1 | 1.96× |
| zero-sum lasso λ = 0.5, n = 10⁵ | 257.9, 7.5 | 163.8, 4.8 | 134.5, 3.8 | 1.92× |
| ℓ1 on [-1, 1], a general normal, n = 10⁵ | 334.3, 7.8 | 203.5, 4.8 | 181.8, 3.9 | 1.84× |
| elastic net on the simplex, n = 10⁵ | 795.0, 15.6 | 247.8, 5.0 | 226.1, 4.1 | 3.52× |

- From the last multiplier, a call takes 3.1 to 4.5 passes instead of 7.5 to
  15.6, and is 1.8 to 3.5 times faster.
- The exact slope saves 0.5 to 1.7 passes per call against the slope of the
  last secant.

**3. FISTA on the zero-sum lasso**: the points at which 500 iterations call the
proximal operator, recorded from one run and replayed for both sides.

| problem | an iteration before #77 (ms) | before #77, each call from zero (μs, passes) | after #77, from the last multiplier (μs, passes) | speedup |
|:-----|---:|---:|---:|---:|
| n = 100, p = 20000 | 0.70 | 62.9, 9.1 | 23.5, 3.1 | 2.68× |
| n = 50, p = 100000 | 5.17 | 294.6, 8.5 | 125.8, 3.4 | 2.34× |

- The calls take 3.1 to 3.4 passes instead of 8.5 to 9.1, and are 2.3 to 2.7
  times faster. Before #77 they took 6 to 9% of an iteration; the products with
  the data matrix take the rest.

**4. The breakpoint search** that #77 proposed (Brucker, 1984; Kiwiel, 2008)
selects the median of the breakpoints left in the bracket, several times. One
selection over the 2n breakpoints of the zero-sum lasso, with the in-place
selection of `IndBallL0`, against the whole search from zero:

| n | fill 2n breakpoints and select their median (μs) | the whole search from zero (μs) |
|---:|---:|---:|
| 1000 | 3.1 | 2.3 |
| 100000 | 1001.2 | 254.3 |

- One selection costs 1.3 to 4 times as much as the whole search, so the
  breakpoint search was not implemented.

**5. The threshold of the projection onto the simplex**, Condat's pass before
#77 against the current one, on the same inputs:

| case | before #77 (μs) | after #77 (μs) | speedup | difference in τ |
|:-----|---:|---:|---:|---:|
| r = 1, n = 10³ | 0.6 | 0.2 | 2.78× | 1e-15 |
| r = 100, n = 10³ | 5.9 | 1.7 | 3.44× | 2e-15 |
| r = n/100, n = 10³ | 2.3 | 0.7 | 3.26× | 8e-15 |
| r = n/10, n = 10³ | 5.9 | 1.7 | 3.43× | 2e-15 |
| r = n/2, n = 10³ | 8.8 | 2.6 | 3.43× | 5e-16 |
| r = 1, n = 10⁵ | 39.7 | 16.5 | 2.40× | 9e-16 |
| r = 100, n = 10⁵ | 188.1 | 179.4 | 1.05× | 1e-13 |
| r = n/100, n = 10⁵ | 568.9 | 282.0 | 2.02× | 2e-14 |
| r = n/10, n = 10⁵ | 1298.4 | 105.2 | 12.34× | 7e-15 |
| r = n/2, n = 10⁵ | 1394.3 | 89.4 | 15.60× | 2e-14 |

- At least as fast in every case, and 2 to 16 times faster in all but r = 100
  at n = 10⁵, with the same threshold up to 1e-13. `IndSimplex`, `IndBallL1`,
  `NormLinf` and `GroupNormLinf` use it.
- Once the level has risen, most entries fall below it, and Condat's pass
  tests four of them at a time. Entry by entry, the speed of the loop depended
  on where its code landed in memory: six identical copies took 30 or 56.5 μs
  at r = 1 and n = 10⁵, and the pass before #77 36 to 43 μs. With the test of
  four, all six take 16.1 to 16.8 μs (jobs 13371750 and 13371886).

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
