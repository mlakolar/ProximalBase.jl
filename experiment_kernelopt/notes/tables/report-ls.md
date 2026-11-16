| operation | implementation | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Skylake-SP, 1.13 |
|:--|:--|--:|--:|--:|--:|
| Float64 value at a residual, n = 100…100000 | **ProximalBase** (time) | 10–8.38e+03 ns | 11.1–8.43e+03 ns | 13.5–1.94e+04 ns | 15.2–8.73e+03 ns |
|  | bare one-loop sum | 0.96 | 0.96 | 0.97 | 0.98 |
|  | `sum(abs2, r)/2` | 1.17 | 1.18 | 1.58 | 1.23 |
|  | `dot(r, r)/2` | 1.21 | 1.16 | 1.33 | 1.52 |
|  | `nrm2(r)^2/2` | 7.10 | 6.81 | 2.84 | 3.95 |
| Float64 value at a predictor, n = 100…100000 | **ProximalBase** (time) | 12.1–1.73e+04 ns | 13.5–1.75e+04 ns | 30.1–3.62e+04 ns | 20.9–7.03e+04 ns |
|  | bare one-loop sum | 1.00 | 1.01 | 0.91 | 0.97 |
| Float64 value at β (one product with X) | **ProximalBase** (time) | 187–1.01e+05 ns | 177–1.02e+05 ns | 268–2.06e+05 ns | 468–4.37e+05 ns |
|  | bare kernel (no checks) | 0.99 | 0.99 | 0.99 | 0.98 |
|  | ProximalOperators | 1.10 | 1.11 | 1.04 | 1.07 |
| Float64 value and gradient at β | **ProximalBase** (time) | 363–2e+05 ns | 357–2e+05 ns | 559–3.94e+05 ns | 945–8.77e+05 ns |
|  | bare: subtract, `gemv`, sum | 1.00 | 1.00 | 1.00 | 0.99 |
|  | bare: fused subtract and sum | 1.00 | 0.99 | 0.99 | 0.99 |
|  | bare: residual by `gemv` | 1.00 | 1.00 | 0.99 | 1.00 |
|  | ProximalOperators | 1.01 | 1.00 | 1.01 | 1.01 |
| Float64 value and gradient at a predictor | **ProximalBase** (time) | 203–9.88e+04 ns | 197–9.94e+04 ns | 308–1.9e+05 ns | 315–3.69e+05 ns |
|  | bare: subtract, `gemv`, sum | 1.00 | 0.98 | 0.99 | 0.99 |
|  | bare: fused subtract and sum | 0.99 | 0.98 | 0.98 | 0.99 |
| Float64 prox, fixed step, tall X | **ProximalBase** (time) | 545–5.03e+04 ns | 545–5.11e+04 ns | 896–9.45e+04 ns | 1.43e+03–2.2e+05 ns |
|  | cached Cholesky | 2.86 | 2.83 | 2.42 | 1.97 |
|  | ProximalOperators | 3.91 | 3.95 | 3.18 | 3.80 |
| Float64 prox, fixed step, wide X (200×1000) | **ProximalBase** (time) | 4.16e+04 ns | 4.19e+04 ns | 7.91e+04 ns | 1.71e+05 ns |
|  | cached Cholesky | 1.56 | 1.56 | 1.46 | 1.35 |
|  | ProximalOperators | 1.85 | 1.97 | 1.65 | 1.64 |
| Float64 prox, the step changes at every call | **ProximalBase** (time) | 1.09e+03–9.77e+04 ns | 1.1e+03–9.71e+04 ns | 1.71e+03–1.89e+05 ns | 2.87e+03–4.41e+05 ns |
|  | cached Cholesky | 14.93 | 14.70 | 11.36 | 15.26 |
|  | ProximalOperators | 26.01 | 20.26 | 17.85 | 20.28 |
| Float64 construction and first prox | **ProximalBase** (time) | 5.6e+05–1.32e+08 ns | 5.51e+05–1.28e+08 ns | 8.66e+05–1.85e+08 ns | 1.29e+06–2.67e+08 ns |
|  | cached Cholesky | 0.07 | 0.07 | 0.07 | 0.08 |
|  | ProximalOperators | 0.08 | 0.08 | 0.08 | 0.09 |