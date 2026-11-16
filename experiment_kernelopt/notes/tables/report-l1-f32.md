| operation | implementation | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Skylake-SP, 1.13 |
|:--|:--|--:|--:|--:|--:|
| Float32 prox, scalar step, n = 128…16384 | **ProximalBase** (time) | 13.5–1.69e+03 ns | 17.1–1.9e+03 ns | 30.1–3.65e+03 ns | 21.9–2.45e+03 ns |
|  | bare loop, PB's formula | 0.99 | 0.99 | 0.98 | 0.94 |
|  | bare loop, select max/min (A) | 0.82 | 0.70 | 0.81 | 0.85 |
|  | bare loop, A3's formula | 0.66 | 0.57 | 0.65 |  |
|  | bare loop, plain clamp | 0.60 | 0.52 | 0.55 | 0.61 |
|  | ProximalOperators | 1.50 | 1.58 | 1.05 | 1.39 |
| Float32 prox, step per entry | **ProximalBase** (time) | 157–1.9e+04 ns | 210–1.04e+05 ns | 312–3.88e+04 ns | 250–3.05e+04 ns |
|  | bare loop, PB's formula | 0.10 | 0.05 | 0.10 | 0.10 |
|  | bare loop, A3's formula | 0.08 | 0.04 | 0.07 |  |
| Float32 value | **ProximalBase** (time) | 18.6–815 ns | 31.9–1.03e+03 ns | 22.9–1.36e+03 ns | 28.4–1.39e+03 ns |
|  | bare one-loop sum | 0.51 | 0.39 | 0.50 | 0.53 |
|  | `sum(abs, x)` | 0.92 | 0.95 | 0.92 | 0.94 |
|  | BLAS `asum` | 0.98 | 0.70 | 0.70 | 0.75 |
|  | ProximalOperators | 0.99 | 0.70 | 0.70 | 0.75 |
| Float32 value, weights | **ProximalBase** (time) | 17.3–1.44e+03 ns | 133–1.79e+04 ns | 29.1–2.76e+03 ns | 19.5–1.92e+03 ns |
|  | bare one-loop sum | 0.52 | 0.06 | 0.43 | 0.77 |
|  | ProximalOperators | 3.48 | 0.56 | 5.35 | 9.90 |