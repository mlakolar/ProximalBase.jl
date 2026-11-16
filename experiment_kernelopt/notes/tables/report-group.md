| operation | implementation | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Skylake-SP, 1.13 |
|:--|:--|--:|--:|--:|--:|
| Float64 prox, groups of 2, range, allshrink | **ProximalBase** (time) | 1.34e+04 ns | 1.78e+04 ns | 2.23e+04 ns | 2.51e+04 ns |
|  | bare kernel (no checks) | 0.91 | 0.60 | 0.86 | 0.84 |
|  | ProximalOperators | 0.76 | 0.56 | 0.60 | 0.60 |
| Float64 prox, groups of 2, range, sparse | **ProximalBase** (time) | 1.38e+04 ns | 1.66e+04 ns | 1.96e+04 ns | 2.44e+04 ns |
|  | bare kernel (no checks) | 0.76 | 0.59 | 0.77 | 0.77 |
|  | ProximalOperators | 0.77 | 0.61 | 0.67 | 0.60 |
| Float64 prox, groups of 2, perm, allshrink | **ProximalBase** (time) | 1.39e+04 ns | 1.56e+04 ns | 2.72e+04 ns | 2.63e+04 ns |
|  | bare kernel (no checks) | 0.75 | 0.68 | 0.74 | 0.78 |
| Float64 prox, groups of 32, range, allshrink | **ProximalBase** (time) | 1.57e+03 ns | 1.82e+03 ns | 2.4e+03 ns | 2.65e+03 ns |
|  | bare kernel (no checks) | 0.84 | 0.76 | 0.83 | 0.81 |
|  | ProximalOperators | 1.73 | 1.61 | 1.49 | 1.86 |
| Float64 prox, groups of 32, range, sparse | **ProximalBase** (time) | 1.64e+03 ns | 1.79e+03 ns | 2.3e+03 ns | 2.62e+03 ns |
|  | bare kernel (no checks) | 0.71 | 0.69 | 0.76 | 0.69 |
|  | ProximalOperators | 1.66 | 1.67 | 1.49 | 1.85 |
| Float64 prox, groups of 32, perm, allshrink | **ProximalBase** (time) | 4.01e+03 ns | 4.5e+03 ns | 8.75e+03 ns | 7.21e+03 ns |
|  | bare kernel (no checks) | 0.93 | 0.93 | 0.99 | 0.97 |
| Float64 prox, groups of 1024, range, allshrink | **ProximalBase** (time) | 686 ns | 690 ns | 1.45e+03 ns | 1.26e+03 ns |
|  | bare kernel (no checks) | 1.01 | 0.96 | 1.01 | 1.01 |
|  | ProximalOperators | 5.41 | 5.37 | 2.88 | 5.04 |
| Float64 prox, groups of 1024, range, sparse | **ProximalBase** (time) | 740 ns | 755 ns | 1.59e+03 ns | 1.27e+03 ns |
|  | bare kernel (no checks) | 0.90 | 0.87 | 0.89 | 0.87 |
|  | ProximalOperators | 5.01 | 4.90 | 2.62 | 4.99 |
| Float64 valprox, groups of 2, range, allshrink | **ProximalBase** (time) | 1.75e+04 ns | 2.15e+04 ns | 2.82e+04 ns | 2.87e+04 ns |
|  | bare kernel (no checks) | 0.72 | 0.53 | 0.70 | 0.80 |
|  | ProximalOperators | 0.58 | 0.47 | 0.48 | 0.53 |
| Float64 valprox, groups of 2, range, sparse | **ProximalBase** (time) | 1.53e+04 ns | 1.74e+04 ns | 2.11e+04 ns | 2.41e+04 ns |
|  | bare kernel (no checks) | 0.75 | 0.58 | 0.70 | 0.84 |
|  | ProximalOperators | 0.69 | 0.58 | 0.62 | 0.61 |
| Float64 valprox, groups of 32, range, allshrink | **ProximalBase** (time) | 1.9e+03 ns | 2.14e+03 ns | 2.76e+03 ns | 3.03e+03 ns |
|  | bare kernel (no checks) | 0.72 | 0.68 | 0.75 | 0.76 |
|  | ProximalOperators | 1.44 | 1.37 | 1.29 | 1.63 |
| Float64 valprox, groups of 32, range, sparse | **ProximalBase** (time) | 1.52e+03 ns | 2.11e+03 ns | 2.45e+03 ns | 2.77e+03 ns |
|  | bare kernel (no checks) | 0.76 | 0.59 | 0.70 | 0.64 |
|  | ProximalOperators | 1.80 | 1.42 | 1.40 | 1.75 |
| Float64 valprox, groups of 1024, range, allshrink | **ProximalBase** (time) | 709 ns | 694 ns | 1.48e+03 ns | 1.28e+03 ns |
|  | bare kernel (no checks) | 0.98 | 0.97 | 0.99 | 0.99 |
|  | ProximalOperators | 5.25 | 5.34 | 2.82 | 4.96 |
| Float64 valprox, groups of 1024, range, sparse | **ProximalBase** (time) | 754 ns | 757 ns | 1.59e+03 ns | 1.25e+03 ns |
|  | bare kernel (no checks) | 0.88 | 0.87 | 0.89 | 0.87 |
|  | ProximalOperators | 4.92 | 4.89 | 2.62 | 5.05 |
| Float64 value, groups of 2, range, allshrink | **ProximalBase** (time) | 8.3e+03 ns | 9.45e+03 ns | 1.15e+04 ns | 1.24e+04 ns |
|  | bare kernel (no checks) | 0.62 | 0.57 | 0.57 | 0.67 |
|  | ProximalOperators | 0.59 | 0.53 | 0.39 | 0.45 |
| Float64 value, groups of 2, range, sparse | **ProximalBase** (time) | 9.58e+03 ns | 1.13e+04 ns | 1.42e+04 ns | 1.49e+04 ns |
|  | bare kernel (no checks) | 0.54 | 0.46 | 0.45 | 0.55 |
|  | ProximalOperators | 0.51 | 0.44 | 0.30 | 0.37 |
| Float64 value, groups of 32, range, allshrink | **ProximalBase** (time) | 904 ns | 1.1e+03 ns | 1.27e+03 ns | 1.41e+03 ns |
|  | bare kernel (no checks) | 0.70 | 0.63 | 0.72 | 0.70 |
|  | ProximalOperators | 2.12 | 1.68 | 1.68 | 2.07 |
| Float64 value, groups of 32, range, sparse | **ProximalBase** (time) | 1.11e+03 ns | 1.49e+03 ns | 2.3e+03 ns | 2.53e+03 ns |
|  | bare kernel (no checks) | 0.56 | 0.46 | 0.40 | 0.38 |
|  | ProximalOperators | 1.70 | 1.25 | 0.89 | 1.13 |
| Float64 value, groups of 1024, range, allshrink | **ProximalBase** (time) | 298 ns | 299 ns | 407 ns | 412 ns |
|  | bare kernel (no checks) | 0.98 | 0.97 | 0.98 | 0.98 |
|  | ProximalOperators | 11.05 | 10.98 | 8.69 | 12.99 |
| Float64 value, groups of 1024, range, sparse | **ProximalBase** (time) | 418 ns | 415 ns | 630 ns | 597 ns |
|  | bare kernel (no checks) | 0.70 | 0.70 | 0.62 | 0.66 |
|  | ProximalOperators | 7.85 | 7.91 | 5.61 | 8.96 |
| Float64 block_sweep, groups of 2, range, allshrink | **ProximalBase** (time) | 2.27e+04 ns | 2.17e+04 ns | 3.78e+04 ns | 4.14e+04 ns |
|  | bare kernel (no checks) | 0.47 | 0.47 | 0.55 | 0.50 |
| Float64 block_sweep, groups of 2, range, sparse | **ProximalBase** (time) | 2.29e+04 ns | 2.12e+04 ns | 3.38e+04 ns | 3.87e+04 ns |
|  | bare kernel (no checks) | 0.45 | 0.45 | 0.46 | 0.37 |
| Float64 block_sweep, groups of 32, range, allshrink | **ProximalBase** (time) | 2.23e+03 ns | 2.37e+03 ns | 3.39e+03 ns | 3.46e+03 ns |
|  | bare kernel (no checks) | 0.57 | 0.68 | 0.61 | 0.61 |
| Float64 block_sweep, groups of 32, range, sparse | **ProximalBase** (time) | 2.16e+03 ns | 2.17e+03 ns | 3.2e+03 ns | 3.44e+03 ns |
|  | bare kernel (no checks) | 0.56 | 0.64 | 0.55 | 0.51 |
| Float64 block_sweep, groups of 1024, range, allshrink | **ProximalBase** (time) | 692 ns | 704 ns | 1.42e+03 ns | 1.13e+03 ns |
|  | bare kernel (no checks) | 0.99 | 0.99 | 1.00 | 1.01 |
| Float64 block_sweep, groups of 1024, range, sparse | **ProximalBase** (time) | 763 ns | 759 ns | 1.61e+03 ns | 1.38e+03 ns |
|  | bare kernel (no checks) | 0.87 | 0.88 | 0.89 | 0.88 |