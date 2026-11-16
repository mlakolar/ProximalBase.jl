| operation | implementation | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Skylake-SP, 1.13 |
|:--|:--|--:|--:|--:|--:|
| Float64 value, n = 2…10 | **ProximalBase** (time) | 6.02–8.79 ns | 5.75–9 ns | 7.67–11.7 ns | 7.78–12.9 ns |
|  | bare one-loop sum of squares | 0.48 | 0.50 | 0.50 | 0.52 |
|  | `LinearAlgebra.norm` | 1.45 | 1.66 | 1.90 | 1.84 |
|  | BLAS `nrm2` | 1.37 | 1.62 | 0.99 | 0.98 |
|  | three-accumulator pass (LAPACK 3.10 style) | 1.16 | 1.25 | 1.13 | 1.29 |
|  | ProximalOperators | 1.45 | 1.67 | 1.88 | 1.77 |
| Float64 value, n = 32…1024 | **ProximalBase** (time) | 17.5–85.8 ns | 21.7–92.3 ns | 21.8–151 ns | 23.7–114 ns |
|  | bare one-loop sum of squares | 0.48 | 0.43 | 0.38 | 0.51 |
|  | `sqrt(sum(abs2, x))` | 0.85 | 0.91 | 0.90 | 0.90 |
|  | `LinearAlgebra.norm` | 2.95 | 2.57 | 1.29 | 1.69 |
|  | BLAS `nrm2` | 2.91 | 2.54 | 1.27 | 1.66 |
|  | `sqrt(dot(x, x))` | 0.75 | 0.67 | 0.62 | 0.75 |
|  | three-accumulator pass (LAPACK 3.10 style) | 4.09 | 3.49 | 2.82 | 5.01 |
|  | ProximalOperators | 2.95 | 2.57 | 1.29 | 1.69 |
| Float64 value, n = 65536 | **ProximalBase** (time) | 5.63e+03 ns | 7e+03 ns | 1.31e+04 ns | 7.23e+03 ns |
|  | bare one-loop sum of squares | 0.83 | 0.77 | 0.97 | 0.79 |
|  | `LinearAlgebra.norm` | 6.38 | 5.14 | 1.63 | 3.40 |
|  | BLAS `nrm2` | 6.39 | 5.14 | 1.63 | 3.40 |
|  | three-accumulator pass (LAPACK 3.10 style) | 9.58 | 7.71 | 4.43 | 12.13 |
|  | ProximalOperators | 6.38 | 5.14 | 1.63 | 3.40 |
| Float64 prox, n = 32…1024 | **ProximalBase** (time) | 13–152 ns | 14.5–155 ns | 17.8–202 ns | 19.7–223 ns |
|  | bare kernel (no checks) | 0.93 | 0.99 | 0.93 | 0.95 |
|  | `norm`, then broadcast | 3.23 | 3.14 | 1.85 | 1.86 |
|  | `nrm2`, then broadcast | 3.18 | 3.09 | 1.83 | 1.82 |
|  | ProximalOperators | 3.17 | 3.09 | 1.82 | 1.81 |
| Float64 prox of the zero vector, n = 32…1024 | **ProximalBase** (time) | 10.4–169 ns | 11.2–169 ns | 17.4–261 ns | 19.2–220 ns |
|  | bare kernel (no checks) | 0.86 | 0.92 | 0.67 | 0.69 |
|  | ProximalOperators | 3.69 | 3.74 | 1.67 | 1.90 |