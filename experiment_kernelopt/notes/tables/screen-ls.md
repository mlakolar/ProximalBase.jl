| operation | implementation | Zen3-1.13 | Zen3-1.10 | Broadwell-1.13 | Broadwell-1.10 | SkylakeSP-1.13 | SkylakeSP-1.10 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **Float64 value at a residual, n = 100…100000** | ProximalBase | 10–8.38e+03 ns | 11.1–8.43e+03 ns | 13.5–1.94e+04 ns | 26–1.93e+04 ns | 15.2–8.73e+03 ns | 15.9–9.67e+03 ns |
| | k_halfsumsq_simd | 0.96 | 0.96 | 0.97 | 0.87 | 0.98 | 0.95 |
| | k_halfsumsq_base | 1.17 | 1.18 | 1.58 | 1.23 | 1.23 | 1.24 |
| | k_halfsumsq_dot | 1.21 | 1.16 | 1.33 | 1.14 | 1.52 | 1.34 |
| | k_halfsumsq_nrm2 | 7.10 | 6.81 | 2.84 | 2.38 | 3.95 | 3.70 |
| **Float64 value at a predictor, n = 100…100000** | ProximalBase | 12.1–1.73e+04 ns | 13.5–1.75e+04 ns | 30.1–3.62e+04 ns | 21.4–3.64e+04 ns | 20.9–7.03e+04 ns | 23.6–7.03e+04 ns |
| | k_simd | 1.00 | 1.01 | 0.91 | 1.04 | 0.97 | 1.06 |
| **Float64 value at β (one product with X)** | ProximalBase | 187–1.01e+05 ns | 177–1.02e+05 ns | 268–2.06e+05 ns | 267–2.06e+05 ns | 468–4.37e+05 ns | 430–4.36e+05 ns |
| | k_direct | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 | 0.98 |
| | po | 1.10 | 1.11 | 1.04 | 1.04 | 1.07 | 1.06 |
| **Float64 value and gradient at β** | ProximalBase | 363–2e+05 ns | 357–2e+05 ns | 559–3.94e+05 ns | 538–3.95e+05 ns | 945–8.77e+05 ns | 851–8.74e+05 ns |
| | k_three | 1.00 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 |
| | k_fused | 1.00 | 0.99 | 0.99 | 0.99 | 0.99 | 0.98 |
| | k_gemv | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 0.99 |
| | po | 1.01 | 1.00 | 1.01 | 1.00 | 1.01 | 1.01 |
| **Float64 value and gradient at a predictor** | ProximalBase | 203–9.88e+04 ns | 197–9.94e+04 ns | 308–1.9e+05 ns | 304–1.9e+05 ns | 315–3.69e+05 ns | 500–3.69e+05 ns |
| | k_three | 1.00 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 |
| | k_fused | 0.99 | 0.98 | 0.98 | 0.98 | 0.99 | 0.97 |
| **Float64 prox, fixed step, tall X** | ProximalBase | 545–5.03e+04 ns | 545–5.11e+04 ns | 896–9.45e+04 ns | 822–9.46e+04 ns | 1.43e+03–2.2e+05 ns | 1.37e+03–2.2e+05 ns |
| | k_chol | 2.86 | 2.83 | 2.42 | 2.50 | 1.97 | 1.82 |
| | po | 3.91 | 3.95 | 3.18 | 3.31 | 3.80 | 3.98 |
| **Float64 prox, fixed step, wide X (200×1000)** | ProximalBase | 4.16e+04 ns | 4.19e+04 ns | 7.91e+04 ns | 7.93e+04 ns | 1.71e+05 ns | 1.71e+05 ns |
| | k_chol | 1.56 | 1.56 | 1.46 | 1.46 | 1.35 | 1.35 |
| | po | 1.85 | 1.97 | 1.65 | 1.65 | 1.64 | 1.65 |
| **Float64 prox, the step changes at every call** | ProximalBase | 1.09e+03–9.77e+04 ns | 1.1e+03–9.71e+04 ns | 1.71e+03–1.89e+05 ns | 1.71e+03–1.89e+05 ns | 2.87e+03–4.41e+05 ns | 2.72e+03–4.4e+05 ns |
| | k_chol | 14.93 | 14.70 | 11.36 | 11.33 | 15.26 | 14.45 |
| | po | 26.01 | 20.26 | 17.85 | 17.23 | 20.28 | 19.28 |
| **Float64 construction and first prox** | ProximalBase | 5.6e+05–1.32e+08 ns | 5.51e+05–1.28e+08 ns | 8.66e+05–1.85e+08 ns | 8.38e+05–1.79e+08 ns | 1.29e+06–2.67e+08 ns | 1.31e+06–2.68e+08 ns |
| | k_chol | 0.07 | 0.07 | 0.07 | 0.07 | 0.08 | 0.08 |
| | po | 0.08 | 0.08 | 0.08 | 0.08 | 0.09 | 0.09 |
| **Float32 value at a residual, n = 100…100000** | ProximalBase | 7.93–3.43e+03 ns | 9.49–3.48e+03 ns | 10.5–9.71e+03 ns | 12.1–9.7e+03 ns | 11.8–4.36e+03 ns | 13.9–4.84e+03 ns |
| | k_halfsumsq_simd | 0.96 | 0.92 | 0.96 | 0.93 | 0.95 | 0.91 |
| | k_halfsumsq_base | 1.34 | 1.31 | 1.53 | 1.43 | 1.42 | 1.36 |
| | k_halfsumsq_dot | 1.27 | 1.18 | 1.38 | 1.27 | 1.73 | 1.43 |
| | k_halfsumsq_nrm2 | 3.73 | 3.45 | 4.46 | 4.18 | 5.64 | 5.00 |
| **Float32 value at a predictor, n = 100…100000** | ProximalBase | 9.57–8.01e+03 ns | 11.8–8.19e+03 ns | 12.8–1.86e+04 ns | 15.8–1.81e+04 ns | 14.9–1e+04 ns | 18.2–9.49e+03 ns |
| | k_simd | 0.96 | 1.01 | 0.97 | 1.02 | 0.98 | 1.01 |
| **Float32 value at β (one product with X)** | ProximalBase | 132–5.24e+04 ns | 123–5.13e+04 ns | 206–9.46e+04 ns | 170–9.45e+04 ns | 280–1.84e+05 ns | 238–1.84e+05 ns |
| | k_direct | 0.99 | 0.99 | 1.00 | 1.00 | 1.01 | 1.15 |
| | po | 1.04 | 1.07 | 1.05 | 1.06 | 1.06 | 1.07 |
| **Float32 value and gradient at β** | ProximalBase | 269–1.03e+05 ns | 255–1.03e+05 ns | 429–1.92e+05 ns | 383–1.92e+05 ns | 664–3.87e+05 ns | 568–3.68e+05 ns |
| | k_three | 0.99 | 1.00 | 1.01 | 0.98 | 0.99 | 1.04 |
| | k_fused | 0.99 | 0.99 | 0.99 | 0.97 | 0.97 | 1.04 |
| | k_gemv | 0.99 | 1.00 | 1.00 | 0.98 | 1.19 | 1.08 |
| | po | 1.00 | 1.04 | 1.00 | 0.99 | 1.19 | 1.09 |
| **Float32 value and gradient at a predictor** | ProximalBase | 157–5.22e+04 ns | 153–5.21e+04 ns | 236–9.77e+04 ns | 218–9.74e+04 ns | 263–1.84e+05 ns | 406–1.84e+05 ns |
| | k_three | 0.99 | 0.98 | 1.00 | 0.98 | 0.98 | 0.99 |
| | k_fused | 0.99 | 0.98 | 0.98 | 0.98 | 0.97 | 0.98 |
| **Float32 prox, fixed step, tall X** | ProximalBase | 400–2.54e+04 ns | 387–2.53e+04 ns | 645–4.88e+04 ns | 624–4.86e+04 ns | 657–3.32e+04 ns | 988–5.06e+04 ns |
| | k_chol | 6.60 | 6.57 | 5.37 | 5.37 | 5.21 | 3.48 |
| | po | 4.38 | 4.46 | 4.19 | 4.24 | 7.54 | 4.89 |
| **Float32 prox, fixed step, wide X (200×1000)** | ProximalBase | 2.02e+04 ns | 2.04e+04 ns | 3.85e+04 ns | 3.85e+04 ns | 2.16e+04 ns | 2.16e+04 ns |
| | k_chol | 2.40 | 2.36 | 2.20 | 2.15 | 2.82 | 2.84 |
| | po | 1.95 | 1.93 | 1.94 | 1.95 | 3.42 | 3.41 |
| **Float32 prox, the step changes at every call** | ProximalBase | 807–5.12e+04 ns | 773–5.02e+04 ns | 1.29e+03–9.78e+04 ns | 1.18e+03–9.74e+04 ns | 1.33e+03–6.67e+04 ns | 2e+03–1.01e+05 ns |
| | k_chol | 19.18 | 19.42 | 15.17 | 15.43 | 44.46 | 32.44 |
| | po | 22.33 | 22.49 | 20.19 | 20.65 | 54.52 | 38.90 |
| **Float32 construction and first prox** | ProximalBase | 4.03e+05–7.87e+07 ns | 3.99e+05–7.54e+07 ns | 6.2e+05–1.07e+08 ns | 6.05e+05–1.04e+08 ns | 9.76e+05–1.42e+08 ns | 9.49e+05–1.45e+08 ns |
| | k_chol | 0.07 | 0.07 | 0.06 | 0.07 | 0.07 | 0.07 |
| | po | 0.07 | 0.07 | 0.07 | 0.07 | 0.08 | 0.08 |
