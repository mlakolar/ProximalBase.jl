| operation | implementation | B13 | B10 | S13 | S10 | Z13shared | Z10shared |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **Float64 value at a residual, n = 100…100000** | ProximalBase | 20.8–1.94e+04 ns | 27.6–1.94e+04 ns | 15.2–8.73e+03 ns | 15.9–9.67e+03 ns | 9.99–1.16e+04 ns | 11.1–1.22e+04 ns |
| | k_halfsumsq_simd | 0.94 | 0.81 | 0.98 | 0.95 | 0.96 | 0.96 |
| | k_halfsumsq_base | 1.21 | 1.12 | 1.23 | 1.24 | 1.14 | 1.14 |
| | k_halfsumsq_dot | 1.19 | 1.11 | 1.52 | 1.34 | 1.19 | 1.14 |
| | k_halfsumsq_nrm2 | 2.49 | 2.25 | 3.95 | 3.70 | 6.54 | 6.22 |
| **Float64 value at a predictor, n = 100…100000** | ProximalBase | 18.7–3.69e+04 ns | 19.8–3.7e+04 ns | 20.9–7.03e+04 ns | 23.6–7.03e+04 ns | 12.3–2.42e+04 ns | 13.8–2.53e+04 ns |
| | k_simd | 0.98 | 1.00 | 0.97 | 1.06 | 0.99 | 1.01 |
| **Float64 value at β (one product with X)** | ProximalBase | 264–2.05e+05 ns | 267–2.06e+05 ns | 468–4.37e+05 ns | 430–4.36e+05 ns | 186–1.26e+05 ns | 173–1.3e+05 ns |
| | k_direct | 0.99 | 1.00 | 0.98 | 0.98 | 0.98 | 1.00 |
| | po | 1.04 | 1.04 | 1.07 | 1.06 | 1.09 | 1.10 |
| **Float64 value and gradient at β** | ProximalBase | 552–3.95e+05 ns | 539–3.95e+05 ns | 945–8.77e+05 ns | 851–8.74e+05 ns | 364–2.48e+05 ns | 355–2.63e+05 ns |
| | k_three | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 | 0.99 |
| | k_fused | 0.99 | 0.99 | 0.99 | 0.98 | 1.00 | 0.99 |
| | k_gemv | 0.99 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 |
| | po | 1.00 | 1.00 | 1.01 | 1.01 | 1.02 | 1.00 |
| **Float64 value and gradient at a predictor** | ProximalBase | 308–1.9e+05 ns | 300–1.9e+05 ns | 315–3.69e+05 ns | 500–3.69e+05 ns | 199–1.24e+05 ns | 195–1.29e+05 ns |
| | k_three | 0.99 | 0.99 | 0.99 | 0.99 | 1.00 | 0.98 |
| | k_fused | 0.98 | 0.98 | 0.99 | 0.97 | 0.99 | 0.99 |
| **Float64 prox, fixed step, tall X** | ProximalBase | 867–9.45e+04 ns | 863–9.45e+04 ns | 1.43e+03–2.2e+05 ns | 1.37e+03–2.2e+05 ns | 543–6.2e+04 ns | 547–6.57e+04 ns |
| | k_chol | 2.44 | 2.45 | 1.97 | 1.82 | 2.76 | 2.69 |
| | po | 3.20 | 3.26 | 3.80 | 3.98 | 4.17 | 4.19 |
| **Float64 prox, fixed step, wide X (200×1000)** | ProximalBase | 7.92e+04 ns | 7.93e+04 ns | 1.71e+05 ns | 1.71e+05 ns | 5.07e+04 ns | 5.32e+04 ns |
| | k_chol | 1.46 | 1.45 | 1.35 | 1.35 | 1.41 | 1.38 |
| | po | 1.65 | 1.66 | 1.64 | 1.65 | 1.67 | 1.68 |
| **Float64 prox, the step changes at every call** | ProximalBase | 1.71e+03–1.89e+05 ns | 1.79e+03–1.89e+05 ns | 2.87e+03–4.41e+05 ns | 2.72e+03–4.4e+05 ns | 1.1e+03–1.24e+05 ns | 1.1e+03–1.31e+05 ns |
| | k_chol | 11.35 | 11.20 | 15.26 | 14.45 | 13.42 | 12.31 |
| | po | 18.09 | 16.90 | 20.28 | 19.28 | 23.54 | 17.63 |
| **Float64 construction and first prox** | ProximalBase | 8.61e+05–1.85e+08 ns | 8.37e+05–1.79e+08 ns | 1.29e+06–2.67e+08 ns | 1.31e+06–2.68e+08 ns | 5.65e+05–1.36e+08 ns | 5.53e+05–1.34e+08 ns |
| | k_chol | 0.07 | 0.07 | 0.08 | 0.08 | 0.07 | 0.07 |
| | po | 0.08 | 0.08 | 0.09 | 0.09 | 0.08 | 0.08 |
| **Float32 value at a residual, n = 100…100000** | ProximalBase | 10.5–9.71e+03 ns | 13–9.71e+03 ns | 11.8–4.36e+03 ns | 13.9–4.84e+03 ns | 7.96–3.9e+03 ns | 9.5–3.78e+03 ns |
| | k_halfsumsq_simd | 0.95 | 0.93 | 0.95 | 0.91 | 0.96 | 0.93 |
| | k_halfsumsq_base | 1.47 | 1.42 | 1.42 | 1.36 | 1.31 | 1.29 |
| | k_halfsumsq_dot | 1.40 | 1.32 | 1.73 | 1.43 | 1.27 | 1.14 |
| | k_halfsumsq_nrm2 | 4.52 | 4.11 | 5.64 | 5.00 | 3.63 | 3.35 |
| **Float32 value at a predictor, n = 100…100000** | ProximalBase | 12.8–1.82e+04 ns | 16.3–1.82e+04 ns | 14.9–1e+04 ns | 18.2–9.49e+03 ns | 9.39–1.04e+04 ns | 11.3–1.14e+04 ns |
| | k_simd | 0.97 | 1.05 | 0.98 | 1.01 | 0.97 | 1.01 |
| **Float32 value at β (one product with X)** | ProximalBase | 212–9.47e+04 ns | 194–9.46e+04 ns | 280–1.84e+05 ns | 238–1.84e+05 ns | 131–6.32e+04 ns | 121–6.57e+04 ns |
| | k_direct | 0.99 | 0.99 | 1.01 | 1.15 | 0.99 | 1.00 |
| | po | 1.08 | 1.03 | 1.06 | 1.07 | 1.04 | 1.04 |
| **Float32 value and gradient at β** | ProximalBase | 447–1.93e+05 ns | 384–1.92e+05 ns | 664–3.87e+05 ns | 568–3.68e+05 ns | 270–1.26e+05 ns | 254–1.29e+05 ns |
| | k_three | 1.00 | 1.00 | 0.99 | 1.04 | 1.00 | 0.99 |
| | k_fused | 0.99 | 1.00 | 0.97 | 1.04 | 0.99 | 0.99 |
| | k_gemv | 0.99 | 1.00 | 1.19 | 1.08 | 0.99 | 0.99 |
| | po | 1.00 | 1.07 | 1.19 | 1.09 | 1.00 | 1.00 |
| **Float32 value and gradient at a predictor** | ProximalBase | 246–9.74e+04 ns | 235–9.76e+04 ns | 263–1.84e+05 ns | 406–1.84e+05 ns | 159–6.24e+04 ns | 150–6.41e+04 ns |
| | k_three | 1.00 | 0.99 | 0.98 | 0.99 | 0.99 | 0.98 |
| | k_fused | 0.99 | 0.99 | 0.97 | 0.98 | 0.99 | 0.99 |
| **Float32 prox, fixed step, tall X** | ProximalBase | 668–4.89e+04 ns | 618–4.87e+04 ns | 657–3.32e+04 ns | 988–5.06e+04 ns | 403–3.11e+04 ns | 383–3.22e+04 ns |
| | k_chol | 5.33 | 5.38 | 5.21 | 3.48 | 6.16 | 6.13 |
| | po | 4.12 | 4.25 | 7.54 | 4.89 | 4.53 | 4.71 |
| **Float32 prox, fixed step, wide X (200×1000)** | ProximalBase | 3.85e+04 ns | 3.84e+04 ns | 2.16e+04 ns | 2.16e+04 ns | 2.43e+04 ns | 2.57e+04 ns |
| | k_chol | 2.22 | 2.16 | 2.82 | 2.84 | 2.20 | 2.06 |
| | po | 1.95 | 1.95 | 3.42 | 3.41 | 1.90 | 1.81 |
| **Float32 prox, the step changes at every call** | ProximalBase | 1.33e+03–9.76e+04 ns | 1.23e+03–9.76e+04 ns | 1.33e+03–6.67e+04 ns | 2e+03–1.01e+05 ns | 824–6.24e+04 ns | 775–6.39e+04 ns |
| | k_chol | 15.09 | 15.27 | 44.46 | 32.44 | 17.71 | 17.55 |
| | po | 20.06 | 20.55 | 54.52 | 38.90 | 23.40 | 22.85 |
| **Float32 construction and first prox** | ProximalBase | 6.21e+05–1.07e+08 ns | 6.09e+05–1.04e+08 ns | 9.76e+05–1.42e+08 ns | 9.49e+05–1.45e+08 ns | 4.05e+05–8.01e+07 ns | 4.05e+05–7.63e+07 ns |
| | k_chol | 0.06 | 0.06 | 0.07 | 0.07 | 0.07 | 0.07 |
| | po | 0.07 | 0.07 | 0.08 | 0.08 | 0.07 | 0.07 |
