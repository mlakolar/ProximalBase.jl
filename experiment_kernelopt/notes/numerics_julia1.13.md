
### Soft thresholding, Float64: 16 values × 7 thresholds

| implementation | wrong values | NaN not propagated | zero results returned as -0 | first difference |
|---|---|---|---|---|
| PB base prox! | 0 | 0 | 0 |  |
| PB A prox! | 0 | 0 | 0 |  |
| PB B prox! | 0 | 0 | 0 |  |
| PB C prox! | 0 | 0 | 0 |  |
| PB C2 prox! | 0 | 0 | 0 |  |
| PB D prox! | 0 | 0 | 0 |  |
| PB P155 prox! | 0 | 0 | 0 |  |
| PB F prox! | 0 | 0 | 0 |  |
| ProximalOperators prox! | 0 | 0 | 0 |  |
| kernel st_maxmin | 0 | 0 | 0 |  |
| kernel st_selmaxmin | 0 | 0 | 0 |  |
| kernel st_clamp | 0 | 0 | 0 |  |
| kernel st_copysign | 0 | 0 | 22 |  |
| kernel st_ifelse | 0 | 9 | 0 | f(Inf, Inf) = 0.0, exact NaN |
| kernel st_po | 0 | 0 | 0 |  |

### Soft thresholding, Float32: 16 values × 7 thresholds

| implementation | wrong values | NaN not propagated | zero results returned as -0 | first difference |
|---|---|---|---|---|
| PB base prox! | 0 | 0 | 0 |  |
| PB A prox! | 0 | 0 | 0 |  |
| PB B prox! | 0 | 0 | 0 |  |
| PB C prox! | 0 | 0 | 0 |  |
| PB C2 prox! | 0 | 0 | 0 |  |
| PB D prox! | 0 | 0 | 0 |  |
| PB P155 prox! | 0 | 0 | 0 |  |
| PB F prox! | 0 | 0 | 0 |  |
| ProximalOperators prox! | 0 | 0 | 0 |  |
| kernel st_maxmin | 0 | 0 | 0 |  |
| kernel st_selmaxmin | 0 | 0 | 0 |  |
| kernel st_clamp | 0 | 0 | 0 |  |
| kernel st_copysign | 0 | 0 | 22 |  |
| kernel st_ifelse | 0 | 9 | 0 | f(Inf, Inf) = 0.0, exact NaN |
| kernel st_po | 0 | 0 | 0 |  |

### Euclidean norm, Float64 (error in eps relative to the exact norm; ok = correctly rounded)

| implementation | ordinary n=1000 | zero vector | entries near sqrt(floatmax)·4 (squares overflow) | entries floatmax/4 | norm overflows (floatmax entries) | entries sqrt(floatmin)/4 (squares underflow) | subnormal entries | one huge, many tiny | mixed 1 and tiny | NaN entry | Inf entry | Inf and NaN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PB base NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB A NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB B NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB C NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB C2 NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB D NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB P155 NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB F NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| ProximalOperators NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| kernel norm_simd | ok | ok | Inf | Inf | ok | ok | 0 (exact 5.93e-323) | Inf | ok | ok | ok | ok |
| kernel norm_la | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| kernel norm_blas | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| kernel norm_dot | ok | ok | Inf | Inf | ok | ok | 0 (exact 5.93e-323) | Inf | ok | ok | ok | ok |
| kernel norm_scaled | ok | ok | ok | ok | ok | ok | Inf | ok | ok | ok | ok | ok |
| kernel norm_blue | 1.55 eps | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |

### ℓ2 proximal operator, Float64 (componentwise error in eps of the exact component)

| implementation | ordinary n=1000, c = ‖x‖/2 | ordinary n=1000, c = 2‖x‖ | zero vector | entries near sqrt(floatmax)·4 (squares overflow), c = ‖x‖/2 | entries near sqrt(floatmax)·4 (squares overflow), c = 2‖x‖ | entries floatmax/4, c = ‖x‖/2 | entries floatmax/4, c = 2‖x‖ | entries sqrt(floatmin)/4 (squares underflow), c = ‖x‖/2 | entries sqrt(floatmin)/4 (squares underflow), c = 2‖x‖ | subnormal entries, c = ‖x‖/2 | subnormal entries, c = 2‖x‖ | one huge, many tiny, c = ‖x‖/2 | one huge, many tiny, c = 2‖x‖ | mixed 1 and tiny, c = ‖x‖/2 | mixed 1 and tiny, c = 2‖x‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PB base prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB A prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB B prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB C prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB C2 prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB D prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB P155 prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB F prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| ProximalOperators prox! | ok | -0 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| kernel direct (plain sum of squares) | ok | ok | ok | ≤ 4.5e+15 eps | nonzero 5.363123171977038e154 | ≤ 4.5e+15 eps | nonzero 4.4942328371557893e307 | ok | ok | 0 (exact 9.88e-324) | ok | ≤ 4.5e+15 eps | nonzero 5.363123171977038e154; nonzero 3.7291703656001034e-155 | ok | ok |
| kernel norm-then-broadcast | ok | -0 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |

### Euclidean norm, Float32 (error in eps relative to the exact norm; ok = correctly rounded)

| implementation | ordinary n=1000 | zero vector | entries near sqrt(floatmax)·4 (squares overflow) | entries floatmax/4 | norm overflows (floatmax entries) | entries sqrt(floatmin)/4 (squares underflow) | subnormal entries | one huge, many tiny | mixed 1 and tiny | NaN entry | Inf entry | Inf and NaN |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PB base NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB A NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB B NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB C NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB C2 NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB D NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB P155 NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB F NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| ProximalOperators NormL2 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| kernel norm_simd | ok | ok | Inf | Inf | ok | ok | 0 (exact 1.68e-44) | Inf | ok | ok | ok | ok |
| kernel norm_la | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| kernel norm_blas | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| kernel norm_dot | ok | ok | Inf | Inf | ok | ok | 0 (exact 1.68e-44) | Inf | ok | ok | ok | ok |
| kernel norm_scaled | ok | ok | ok | ok | ok | ok | Inf | ok | ok | ok | ok | ok |
| kernel norm_blue | 2.16 eps | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |

### ℓ2 proximal operator, Float32 (componentwise error in eps of the exact component)

| implementation | ordinary n=1000, c = ‖x‖/2 | ordinary n=1000, c = 2‖x‖ | zero vector | entries near sqrt(floatmax)·4 (squares overflow), c = ‖x‖/2 | entries near sqrt(floatmax)·4 (squares overflow), c = 2‖x‖ | entries floatmax/4, c = ‖x‖/2 | entries floatmax/4, c = 2‖x‖ | entries sqrt(floatmin)/4 (squares underflow), c = ‖x‖/2 | entries sqrt(floatmin)/4 (squares underflow), c = 2‖x‖ | subnormal entries, c = ‖x‖/2 | subnormal entries, c = 2‖x‖ | one huge, many tiny, c = ‖x‖/2 | one huge, many tiny, c = 2‖x‖ | mixed 1 and tiny, c = ‖x‖/2 | mixed 1 and tiny, c = 2‖x‖ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PB base prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB A prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB B prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB C prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB C2 prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB D prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB P155 prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| PB F prox! | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| ProximalOperators prox! | ok | -0 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |
| kernel direct (plain sum of squares) | ok | ok | ok | ≤ 8.39e+06 eps | nonzero 7.378697e19 | ≤ 8.39e+06 eps | nonzero 8.5070587e37 | ok | ok | 0 (exact 2.1e-45) | ok | ≤ 8.39e+06 eps | nonzero 7.378697e19; nonzero 2.7105054e-20 | ok | ok |
| kernel norm-then-broadcast | ok | -0 | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok | ok |

### ‖r‖²/2, Float64

| implementation | ordinary n=1000 | zero | squares overflow, value finite? | squares overflow | squares underflow | subnormal |
|---|---|---|---|---|---|---|
| PB base value_from_residual | ok | ok | ok | ok | ok | ok |
| PB A value_from_residual | ok | ok | ok | ok | ok | ok |
| PB B value_from_residual | ok | ok | ok | ok | ok | ok |
| PB C value_from_residual | ok | ok | ok | ok | ok | ok |
| PB C2 value_from_residual | ok | ok | ok | ok | ok | ok |
| PB D value_from_residual | ok | ok | ok | ok | ok | ok |
| PB P155 value_from_residual | ok | ok | ok | ok | ok | ok |
| PB F value_from_residual | ok | ok | ok | ok | ok | ok |
| kernel halfsumsq_simd | ok | ok | ok | ok | ok | ok |
| kernel halfsumsq_base | ok | ok | ok | ok | ok | ok |
| kernel halfsumsq_dot | ok | ok | ok | ok | ok | ok |
| kernel halfsumsq_nrm2 | ok | ok | ok | ok | ok | ok |

### Least-squares proximal operator, Float64, 60 × 12: relative error ‖β − β*‖/‖β*‖ (eps = 2.2e-16)

| implementation | cond 1e+02, γ = 1e+00 | cond 1e+02, γ = 1e+06 | cond 1e+06, γ = 1e+00 | cond 1e+06, γ = 1e+06 | cond 1e+10, γ = 1e+00 | cond 1e+10, γ = 1e+06 | cond 1e+14, γ = 1e+00 | cond 1e+14, γ = 1e+06 |
|---|---|---|---|---|---|---|---|---|
| PB base prox! (thin SVD) | 8.4e-16 | 5.7e-15 | 4.0e-16 | 8.9e-14 | 4.5e-16 | 2.3e-13 | 1.0e-15 | 2.9e-13 |
| PB A prox! (thin SVD) | 8.4e-16 | 5.7e-15 | 4.0e-16 | 8.9e-14 | 4.5e-16 | 2.3e-13 | 1.0e-15 | 2.9e-13 |
| PB B prox! (thin SVD) | 8.4e-16 | 5.7e-15 | 4.0e-16 | 8.9e-14 | 4.5e-16 | 2.3e-13 | 1.0e-15 | 2.9e-13 |
| PB C prox! (thin SVD) | 8.4e-16 | 5.7e-15 | 4.0e-16 | 8.9e-14 | 4.5e-16 | 2.3e-13 | 1.0e-15 | 2.9e-13 |
| PB C2 prox! (thin SVD) | 8.4e-16 | 5.7e-15 | 4.0e-16 | 8.9e-14 | 4.5e-16 | 2.3e-13 | 1.0e-15 | 2.9e-13 |
| PB D prox! (thin SVD) | 8.4e-16 | 5.7e-15 | 4.0e-16 | 8.9e-14 | 4.5e-16 | 2.3e-13 | 1.0e-15 | 2.9e-13 |
| PB P155 prox! (thin SVD) | 8.4e-16 | 5.7e-15 | 4.0e-16 | 8.9e-14 | 4.5e-16 | 2.3e-13 | 1.0e-15 | 2.9e-13 |
| PB F prox! (thin SVD) | 8.4e-16 | 5.7e-15 | 4.0e-16 | 8.9e-14 | 4.5e-16 | 2.3e-13 | 1.0e-15 | 2.9e-13 |
| kernel Cholesky of X'X + I/γ | 1.8e-16 | 1.1e-13 | 2.5e-16 | 5.3e-11 | 2.1e-16 | 2.2e-11 | 2.9e-16 | 6.4e-11 |
| ProximalOperators LeastSquares | 1.4e-16 | 1.1e-13 | 1.7e-16 | 5.3e-11 | 1.7e-16 | 2.2e-11 | 3.2e-16 | 6.4e-11 |

### ‖r‖²/2, Float32

| implementation | ordinary n=1000 | zero | squares overflow, value finite? | squares overflow | squares underflow | subnormal |
|---|---|---|---|---|---|---|
| PB base value_from_residual | ok | ok | ok | ok | ok | ok |
| PB A value_from_residual | ok | ok | ok | ok | ok | ok |
| PB B value_from_residual | ok | ok | ok | ok | ok | ok |
| PB C value_from_residual | ok | ok | ok | ok | ok | ok |
| PB C2 value_from_residual | ok | ok | ok | ok | ok | ok |
| PB D value_from_residual | ok | ok | ok | ok | ok | ok |
| PB P155 value_from_residual | ok | ok | ok | ok | ok | ok |
| PB F value_from_residual | ok | ok | ok | ok | ok | ok |
| kernel halfsumsq_simd | ok | ok | ok | ok | ok | ok |
| kernel halfsumsq_base | ok | ok | ok | ok | ok | ok |
| kernel halfsumsq_dot | ok | ok | ok | ok | ok | ok |
| kernel halfsumsq_nrm2 | ok | ok | 0.5 eps | ok | ok | ok |

### Least-squares proximal operator, Float32, 60 × 12: relative error ‖β − β*‖/‖β*‖ (eps = 1.2e-07)

| implementation | cond 1e+01, γ = 1e+00 | cond 1e+01, γ = 1e+06 | cond 1e+03, γ = 1e+00 | cond 1e+03, γ = 1e+06 | cond 1e+05, γ = 1e+00 | cond 1e+05, γ = 1e+06 | cond 1e+07, γ = 1e+00 | cond 1e+07, γ = 1e+06 |
|---|---|---|---|---|---|---|---|---|
| PB base prox! (thin SVD) | 5.0e-07 | 9.8e-07 | 4.6e-07 | 2.0e-05 | 3.2e-07 | 2.8e-05 | 2.7e-07 | 9.4e-05 |
| PB A prox! (thin SVD) | 5.0e-07 | 9.8e-07 | 4.6e-07 | 2.0e-05 | 3.2e-07 | 2.8e-05 | 2.7e-07 | 9.4e-05 |
| PB B prox! (thin SVD) | 5.0e-07 | 9.8e-07 | 4.6e-07 | 2.0e-05 | 3.2e-07 | 2.8e-05 | 2.7e-07 | 9.4e-05 |
| PB C prox! (thin SVD) | 5.0e-07 | 9.8e-07 | 4.6e-07 | 2.0e-05 | 3.2e-07 | 2.8e-05 | 2.7e-07 | 9.4e-05 |
| PB C2 prox! (thin SVD) | 5.0e-07 | 9.8e-07 | 4.6e-07 | 2.0e-05 | 3.2e-07 | 2.8e-05 | 2.7e-07 | 9.4e-05 |
| PB D prox! (thin SVD) | 5.0e-07 | 9.8e-07 | 4.6e-07 | 2.0e-05 | 3.2e-07 | 2.8e-05 | 2.7e-07 | 9.4e-05 |
| PB P155 prox! (thin SVD) | 5.0e-07 | 9.8e-07 | 4.6e-07 | 2.0e-05 | 3.2e-07 | 2.8e-05 | 2.7e-07 | 9.4e-05 |
| PB F prox! (thin SVD) | 5.0e-07 | 9.8e-07 | 4.6e-07 | 2.0e-05 | 3.2e-07 | 2.8e-05 | 2.7e-07 | 9.4e-05 |
| kernel Cholesky of X'X + I/γ | 1.6e-07 | 2.5e-06 | 1.2e-07 | 3.5e-03 | 1.6e-07 | 2.9e-02 | 7.3e-08 | 1.8e-02 |
| ProximalOperators LeastSquares | 1.5e-07 | 2.5e-06 | 1.3e-07 | 3.5e-03 | 1.7e-07 | 2.9e-02 | 1.5e-07 | 1.8e-02 |
