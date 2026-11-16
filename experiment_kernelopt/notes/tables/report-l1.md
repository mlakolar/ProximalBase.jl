| operation | implementation | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Skylake-SP, 1.13 |
|:--|:--|--:|--:|--:|--:|
| Float64 prox, scalar step, n = 128…16384 | **ProximalBase** (time) | 26.6–3.38e+03 ns | 31.4–3.83e+03 ns | 58.3–7.32e+03 ns | 36.3–4.88e+03 ns |
|  | bare loop, PB's formula | 1.00 | 1.00 | 0.99 | 0.98 |
|  | bare loop, select max/min (A) | 0.83 | 0.72 | 0.82 | 0.85 |
|  | bare loop, A3's formula | 0.67 | 0.59 | 0.66 |  |
|  | bare loop, plain clamp | 0.56 | 0.50 | 0.56 | 0.62 |
|  | ProximalOperators | 1.20 | 1.56 | 1.02 | 1.36 |
| Float64 prox, weights | **ProximalBase** (time) | 28.2–3.42e+03 ns | 37.5–4.64e+03 ns | 60.5–8.25e+03 ns | 45.5–5.98e+03 ns |
|  | bare loop, PB's formula | 0.99 | 0.98 | 1.00 | 1.00 |
|  | bare loop, A3's formula | 0.81 | 0.62 | 0.77 |  |
|  | ProximalOperators | 4.07 | 5.21 | 6.80 | 6.12 |
| Float64 prox, step per entry | **ProximalBase** (time) | 149–1.92e+04 ns | 226–2.76e+04 ns | 311–3.88e+04 ns | 250–3.05e+04 ns |
|  | bare loop, PB's formula | 0.18 | 0.16 | 0.20 | 0.18 |
|  | bare loop, A3's formula | 0.15 | 0.10 | 0.15 |  |
| Float64 value | **ProximalBase** (time) | 19–1.3e+03 ns | 22.2–1.35e+03 ns | 26.3–2.46e+03 ns | 30–2.13e+03 ns |
|  | bare one-loop sum | 0.67 | 0.62 | 0.60 | 0.70 |
|  | `sum(abs, x)` | 0.92 | 0.94 | 0.94 | 0.95 |
|  | BLAS `asum` | 1.23 | 1.13 | 0.72 | 0.85 |
|  | ProximalOperators | 1.27 | 1.14 | 0.75 | 0.87 |
| Float64 value, weights | **ProximalBase** (time) | 27.6–2.87e+03 ns | 133–1.79e+04 ns | 49.3–5.79e+03 ns | 31.2–3.82e+03 ns |
|  | bare one-loop sum | 0.52 | 0.10 | 0.44 | 0.79 |
|  | ProximalOperators | 5.38 | 1.08 | 3.81 | 5.73 |
| Float64 value and prox | **ProximalBase** (time) | 38.2–4.54e+03 ns | 47.8–4.95e+03 ns | 77.3–9.53e+03 ns | 59.2–6.94e+03 ns |
|  | bare, two passes | 0.95 | 0.94 | 0.95 | 0.92 |
|  | bare, one fused pass | 0.90 | 0.91 | 0.91 | 0.84 |
|  | bare fused pass, plain clamp | 0.58 | 0.49 | 0.54 | 0.57 |
|  | ProximalOperators | 0.89 | 1.13 | 0.80 | 0.92 |
| Float64 prox, n = 2^20 (out of cache) | **ProximalBase** (time) | 2.17e+05 ns | 2.47e+05 ns | 5.69e+05 ns | 8.83e+05 ns |
|  | bare loop, PB's formula | 1.00 | 1.00 | 1.00 | 1.00 |
|  | bare loop, A3's formula | 0.84 | 0.75 | 0.98 |  |
|  | ProximalOperators | 1.13 | 1.57 | 1.02 | 1.02 |