| operation | implementation | B13 | B10 | S13 | S10 | Z13shared | Z10shared |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **Float64 prox, groups of 2, range, allshrink** | ProximalBase | 2.17e+04 ns | 2.48e+04 ns | 2.51e+04 ns | 2.58e+04 ns | 1.23e+04 ns | 1.79e+04 ns |
| | k_direct | 0.88 | 0.76 | 0.84 | 0.80 | 0.83 | 0.60 |
| | po | 0.62 | 0.57 | 0.60 | 0.58 | 0.83 | 0.55 |
| **Float64 prox, groups of 2, range, sparse** | ProximalBase | 1.97e+04 ns | 2.15e+04 ns | 2.44e+04 ns | 2.45e+04 ns | 1.39e+04 ns | 1.67e+04 ns |
| | k_direct | 0.77 | 0.68 | 0.77 | 0.79 | 0.73 | 0.60 |
| | po | 0.67 | 0.63 | 0.60 | 0.59 | 0.73 | 0.61 |
| **Float64 prox, groups of 2, perm, allshrink** | ProximalBase | 2.64e+04 ns | 2.67e+04 ns | 2.63e+04 ns | 2.87e+04 ns | 1.43e+04 ns | 1.54e+04 ns |
| | k_direct | 0.78 | 0.80 | 0.78 | 0.78 | 0.77 | 0.71 |
| **Float64 prox, groups of 5, range, allshrink** | ProximalBase | 1.19e+04 ns | 1.39e+04 ns | 1.25e+04 ns | 1.17e+04 ns | 7.99e+03 ns | 8.92e+03 ns |
| | k_direct | 0.89 | 0.67 | 0.88 | 0.76 | 0.81 | 0.68 |
| | po | 0.80 | 0.62 | 0.77 | 0.75 | 0.75 | 0.64 |
| **Float64 prox, groups of 5, range, sparse** | ProximalBase | 1.12e+04 ns | 1.3e+04 ns | 1.11e+04 ns | 1.29e+04 ns | 7.24e+03 ns | 8.12e+03 ns |
| | k_direct | 0.80 | 0.66 | 0.91 | 0.58 | 0.83 | 0.70 |
| | po | 0.79 | 0.64 | 0.84 | 0.67 | 0.83 | 0.71 |
| **Float64 prox, groups of 5, perm, allshrink** | ProximalBase | 1.32e+04 ns | 1.22e+04 ns | 1.38e+04 ns | 1.25e+04 ns | 1.02e+04 ns | 9.43e+03 ns |
| | k_direct | 0.80 | 0.83 | 0.82 | 0.80 | 0.81 | 0.78 |
| **Float64 prox, groups of 32, range, allshrink** | ProximalBase | 2.41e+03 ns | 2.59e+03 ns | 2.65e+03 ns | 2.76e+03 ns | 1.63e+03 ns | 1.93e+03 ns |
| | k_direct | 0.83 | 0.77 | 0.81 | 0.78 | 0.81 | 0.75 |
| | po | 1.48 | 1.44 | 1.86 | 1.82 | 1.69 | 1.52 |
| **Float64 prox, groups of 32, range, sparse** | ProximalBase | 2.35e+03 ns | 2.68e+03 ns | 2.62e+03 ns | 2.89e+03 ns | 1.46e+03 ns | 1.8e+03 ns |
| | k_direct | 0.75 | 0.64 | 0.69 | 0.61 | 0.83 | 0.69 |
| | po | 1.47 | 1.34 | 1.85 | 1.71 | 1.87 | 1.67 |
| **Float64 prox, groups of 32, perm, allshrink** | ProximalBase | 8.77e+03 ns | 8.74e+03 ns | 7.21e+03 ns | 7.89e+03 ns | 3.97e+03 ns | 4.56e+03 ns |
| | k_direct | 0.99 | 0.99 | 0.97 | 0.94 | 0.97 | 0.96 |
| **Float64 prox, groups of 1024, range, allshrink** | ProximalBase | 1.46e+03 ns | 1.42e+03 ns | 1.26e+03 ns | 1.24e+03 ns | 701 ns | 734 ns |
| | k_direct | 1.01 | 1.02 | 1.01 | 1.01 | 0.98 | 0.97 |
| | po | 2.85 | 2.94 | 5.04 | 5.11 | 5.30 | 5.10 |
| **Float64 prox, groups of 1024, range, sparse** | ProximalBase | 1.61e+03 ns | 1.59e+03 ns | 1.27e+03 ns | 1.2e+03 ns | 758 ns | 757 ns |
| | k_direct | 0.88 | 0.89 | 0.87 | 0.90 | 0.90 | 0.87 |
| | po | 2.59 | 2.63 | 4.99 | 5.25 | 4.96 | 4.91 |
| **Float64 prox, groups of 1024, perm, allshrink** | ProximalBase | 9.71e+03 ns | 9.58e+03 ns | 7.54e+03 ns | 7.55e+03 ns | 4.42e+03 ns | 4.49e+03 ns |
| | k_direct | 0.98 | 0.99 | 0.99 | 0.99 | 0.98 | 1.00 |
| **Float64 valprox, groups of 2, range, allshrink** | ProximalBase | 2.73e+04 ns | 3.1e+04 ns | 2.87e+04 ns | 3.14e+04 ns | 1.76e+04 ns | 2.33e+04 ns |
| | k_direct | 0.72 | 0.61 | 0.80 | 0.67 | 0.66 | 0.49 |
| | po | 0.49 | 0.46 | 0.53 | 0.48 | 0.58 | 0.43 |
| **Float64 valprox, groups of 2, range, sparse** | ProximalBase | 2.11e+04 ns | 2.22e+04 ns | 2.41e+04 ns | 2.45e+04 ns | 1.55e+04 ns | 1.75e+04 ns |
| | k_direct | 0.71 | 0.65 | 0.84 | 0.74 | 0.70 | 0.58 |
| | po | 0.63 | 0.61 | 0.61 | 0.59 | 0.66 | 0.58 |
| **Float64 valprox, groups of 2, perm, allshrink** | ProximalBase | 3.11e+04 ns | 3.18e+04 ns | 3.32e+04 ns | 3.5e+04 ns | 2.09e+04 ns | 2.08e+04 ns |
| | k_direct | 0.69 | 0.69 | 0.64 | 0.71 | 0.53 | 0.54 |
| **Float64 valprox, groups of 5, range, allshrink** | ProximalBase | 1.89e+04 ns | 1.66e+04 ns | 1.4e+04 ns | 1.39e+04 ns | 9.78e+03 ns | 1.02e+04 ns |
| | k_direct | 0.58 | 0.59 | 0.82 | 0.64 | 0.70 | 0.64 |
| | po | 0.51 | 0.52 | 0.68 | 0.63 | 0.61 | 0.56 |
| **Float64 valprox, groups of 5, range, sparse** | ProximalBase | 1.18e+04 ns | 1.33e+04 ns | 1.17e+04 ns | 1.3e+04 ns | 7.94e+03 ns | 8.64e+03 ns |
| | k_direct | 0.74 | 0.64 | 0.87 | 0.60 | 0.78 | 0.67 |
| | po | 0.75 | 0.62 | 0.79 | 0.66 | 0.76 | 0.66 |
| **Float64 valprox, groups of 5, perm, allshrink** | ProximalBase | 1.57e+04 ns | 1.53e+04 ns | 1.78e+04 ns | 1.61e+04 ns | 1.24e+04 ns | 1.12e+04 ns |
| | k_direct | 0.73 | 0.70 | 0.67 | 0.65 | 0.70 | 0.69 |
| **Float64 valprox, groups of 32, range, allshrink** | ProximalBase | 2.81e+03 ns | 2.91e+03 ns | 3.03e+03 ns | 3.15e+03 ns | 1.94e+03 ns | 2.13e+03 ns |
| | k_direct | 0.74 | 0.71 | 0.76 | 0.72 | 0.72 | 0.70 |
| | po | 1.26 | 1.27 | 1.63 | 1.59 | 1.43 | 1.38 |
| **Float64 valprox, groups of 32, range, sparse** | ProximalBase | 2.44e+03 ns | 2.73e+03 ns | 2.77e+03 ns | 2.97e+03 ns | 1.57e+03 ns | 1.85e+03 ns |
| | k_direct | 0.71 | 0.64 | 0.64 | 0.60 | 0.77 | 0.67 |
| | po | 1.41 | 1.30 | 1.75 | 1.66 | 1.73 | 1.61 |
| **Float64 valprox, groups of 32, perm, allshrink** | ProximalBase | 8.99e+03 ns | 9.05e+03 ns | 7.61e+03 ns | 8.03e+03 ns | 4.98e+03 ns | 4.88e+03 ns |
| | k_direct | 0.97 | 0.97 | 0.93 | 0.92 | 0.78 | 0.93 |
| **Float64 valprox, groups of 1024, range, allshrink** | ProximalBase | 1.46e+03 ns | 1.45e+03 ns | 1.28e+03 ns | 1.26e+03 ns | 718 ns | 723 ns |
| | k_direct | 1.01 | 1.01 | 0.99 | 0.98 | 0.96 | 0.98 |
| | po | 2.86 | 2.87 | 4.96 | 5.01 | 5.17 | 5.18 |
| **Float64 valprox, groups of 1024, range, sparse** | ProximalBase | 1.6e+03 ns | 1.59e+03 ns | 1.25e+03 ns | 1.2e+03 ns | 756 ns | 754 ns |
| | k_direct | 0.88 | 0.88 | 0.87 | 0.90 | 0.89 | 0.87 |
| | po | 2.59 | 2.61 | 5.05 | 5.25 | 4.97 | 4.94 |
| **Float64 valprox, groups of 1024, perm, allshrink** | ProximalBase | 9.67e+03 ns | 9.59e+03 ns | 7.55e+03 ns | 7.57e+03 ns | 4.42e+03 ns | 4.5e+03 ns |
| | k_direct | 0.99 | 1.00 | 1.00 | 0.99 | 0.99 | 1.00 |
| **Float64 value, groups of 2, range, allshrink** | ProximalBase | 1.15e+04 ns | 1.28e+04 ns | 1.24e+04 ns | 1.26e+04 ns | 8.29e+03 ns | 9.47e+03 ns |
| | k_direct | 0.57 | 0.51 | 0.67 | 0.54 | 0.63 | 0.55 |
| | po | 0.39 | 0.35 | 0.45 | 0.39 | 0.60 | 0.53 |
| **Float64 value, groups of 2, range, sparse** | ProximalBase | 1.38e+04 ns | 1.42e+04 ns | 1.49e+04 ns | 1.56e+04 ns | 9.63e+03 ns | 1.13e+04 ns |
| | k_direct | 0.46 | 0.45 | 0.55 | 0.42 | 0.54 | 0.46 |
| | po | 0.31 | 0.30 | 0.37 | 0.31 | 0.51 | 0.44 |
| **Float64 value, groups of 2, perm, allshrink** | ProximalBase | 1.37e+04 ns | 1.35e+04 ns | 1.3e+04 ns | 1.33e+04 ns | 9.48e+03 ns | 9.82e+03 ns |
| | k_direct | 0.57 | 0.64 | 0.58 | 0.60 | 0.58 | 0.55 |
| **Float64 value, groups of 5, range, allshrink** | ProximalBase | 6.25e+03 ns | 6.71e+03 ns | 6.5e+03 ns | 6.95e+03 ns | 4.7e+03 ns | 4.99e+03 ns |
| | k_direct | 0.65 | 0.61 | 0.63 | 0.59 | 0.58 | 0.53 |
| | po | 0.46 | 0.42 | 0.55 | 0.46 | 0.54 | 0.43 |
| **Float64 value, groups of 5, range, sparse** | ProximalBase | 7.69e+03 ns | 7.78e+03 ns | 7.89e+03 ns | 8.23e+03 ns | 5.53e+03 ns | 5.99e+03 ns |
| | k_direct | 0.50 | 0.50 | 0.50 | 0.50 | 0.49 | 0.44 |
| | po | 0.37 | 0.36 | 0.46 | 0.38 | 0.46 | 0.36 |
| **Float64 value, groups of 5, perm, allshrink** | ProximalBase | 7.4e+03 ns | 6.6e+03 ns | 7.98e+03 ns | 6.69e+03 ns | 5.99e+03 ns | 5.1e+03 ns |
| | k_direct | 0.68 | 0.71 | 0.64 | 0.69 | 0.59 | 0.60 |
| **Float64 value, groups of 32, range, allshrink** | ProximalBase | 1.29e+03 ns | 1.4e+03 ns | 1.41e+03 ns | 1.51e+03 ns | 907 ns | 1.1e+03 ns |
| | k_direct | 0.72 | 0.67 | 0.70 | 0.65 | 0.68 | 0.63 |
| | po | 1.66 | 1.55 | 2.07 | 1.94 | 2.07 | 1.68 |
| **Float64 value, groups of 32, range, sparse** | ProximalBase | 2.3e+03 ns | 1.75e+03 ns | 2.53e+03 ns | 1.84e+03 ns | 1.12e+03 ns | 1.37e+03 ns |
| | k_direct | 0.40 | 0.51 | 0.38 | 0.53 | 0.56 | 0.50 |
| | po | 0.90 | 1.18 | 1.13 | 1.57 | 1.69 | 1.35 |
| **Float64 value, groups of 32, perm, allshrink** | ProximalBase | 3.13e+03 ns | 3.14e+03 ns | 2.89e+03 ns | 2.79e+03 ns | 1.77e+03 ns | 2.21e+03 ns |
| | k_direct | 0.92 | 0.89 | 0.87 | 0.92 | 0.88 | 0.95 |
| **Float64 value, groups of 1024, range, allshrink** | ProximalBase | 406 ns | 409 ns | 412 ns | 408 ns | 301 ns | 306 ns |
| | k_direct | 0.98 | 0.94 | 0.98 | 0.96 | 0.99 | 0.98 |
| | po | 8.72 | 8.66 | 12.99 | 13.12 | 10.93 | 10.76 |
| **Float64 value, groups of 1024, range, sparse** | ProximalBase | 637 ns | 618 ns | 597 ns | 575 ns | 403 ns | 415 ns |
| | k_direct | 0.62 | 0.63 | 0.66 | 0.69 | 0.73 | 0.71 |
| | po | 5.55 | 5.72 | 8.96 | 9.31 | 8.16 | 7.91 |
| **Float64 value, groups of 1024, perm, allshrink** | ProximalBase | 1.83e+03 ns | 1.83e+03 ns | 1.7e+03 ns | 1.69e+03 ns | 1.25e+03 ns | 1.23e+03 ns |
| | k_direct | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 | 1.01 |
| **Float64 block_sweep, groups of 2, range, allshrink** | ProximalBase | 3.87e+04 ns | 3.63e+04 ns | 4.14e+04 ns | 3.97e+04 ns | 2.27e+04 ns | 2.19e+04 ns |
| | k_direct | 0.55 | 0.57 | 0.50 | 0.52 | 0.49 | 0.47 |
| **Float64 block_sweep, groups of 2, range, sparse** | ProximalBase | 3.41e+04 ns | 3.27e+04 ns | 3.87e+04 ns | 3.42e+04 ns | 2.34e+04 ns | 2.2e+04 ns |
| | k_direct | 0.46 | 0.49 | 0.37 | 0.43 | 0.45 | 0.45 |
| **Float64 block_sweep, groups of 5, range, allshrink** | ProximalBase | 1.6e+04 ns | 1.56e+04 ns | 1.73e+04 ns | 1.66e+04 ns | 1.23e+04 ns | 1.12e+04 ns |
| | k_direct | 0.55 | 0.60 | 0.50 | 0.54 | 0.52 | 0.58 |
| **Float64 block_sweep, groups of 5, range, sparse** | ProximalBase | 1.67e+04 ns | 1.57e+04 ns | 1.6e+04 ns | 1.64e+04 ns | 1.04e+04 ns | 9.75e+03 ns |
| | k_direct | 0.57 | 0.53 | 0.44 | 0.43 | 0.46 | 0.50 |
| **Float64 block_sweep, groups of 32, range, allshrink** | ProximalBase | 3.42e+03 ns | 3.36e+03 ns | 3.46e+03 ns | 3.63e+03 ns | 2.23e+03 ns | 2.42e+03 ns |
| | k_direct | 0.60 | 0.69 | 0.61 | 0.63 | 0.56 | 0.69 |
| **Float64 block_sweep, groups of 32, range, sparse** | ProximalBase | 3.22e+03 ns | 3.33e+03 ns | 3.44e+03 ns | 3.62e+03 ns | 2.23e+03 ns | 2.27e+03 ns |
| | k_direct | 0.56 | 0.57 | 0.51 | 0.52 | 0.52 | 0.66 |
| **Float64 block_sweep, groups of 1024, range, allshrink** | ProximalBase | 1.39e+03 ns | 1.44e+03 ns | 1.13e+03 ns | 1.15e+03 ns | 710 ns | 691 ns |
| | k_direct | 1.00 | 0.99 | 1.01 | 0.99 | 0.97 | 0.98 |
| **Float64 block_sweep, groups of 1024, range, sparse** | ProximalBase | 1.61e+03 ns | 1.67e+03 ns | 1.38e+03 ns | 1.33e+03 ns | 762 ns | 766 ns |
| | k_direct | 0.89 | 0.90 | 0.88 | 0.90 | 0.88 | 0.88 |
| **Float32 prox, groups of 2, range, allshrink** | ProximalBase | 2.09e+04 ns | 2.39e+04 ns | 2.55e+04 ns | 2.55e+04 ns | 1.13e+04 ns | 1.51e+04 ns |
| | k_direct | 0.82 | 0.72 | 0.77 | 0.69 | 0.77 | 0.70 |
| | po | 0.57 | 0.52 | 0.52 | 0.52 | 0.77 | 0.56 |
| **Float32 prox, groups of 2, range, sparse** | ProximalBase | 1.92e+04 ns | 2.08e+04 ns | 2.42e+04 ns | 2.38e+04 ns | 1.3e+04 ns | 1.58e+04 ns |
| | k_direct | 0.71 | 0.72 | 0.76 | 0.70 | 0.76 | 0.63 |
| | po | 0.61 | 0.58 | 0.55 | 0.56 | 0.67 | 0.53 |
| **Float32 prox, groups of 2, perm, allshrink** | ProximalBase | 2.31e+04 ns | 2.44e+04 ns | 2.43e+04 ns | 2.6e+04 ns | 1.31e+04 ns | 1.35e+04 ns |
| | k_direct | 0.80 | 0.77 | 0.77 | 0.75 | 0.66 | 0.65 |
| **Float32 prox, groups of 5, range, allshrink** | ProximalBase | 9.16e+03 ns | 1.14e+04 ns | 1.02e+04 ns | 1.12e+04 ns | 7e+03 ns | 8.1e+03 ns |
| | k_direct | 0.80 | 0.70 | 0.77 | 0.70 | 0.80 | 0.74 |
| | po | 0.74 | 0.65 | 0.78 | 0.73 | 0.74 | 0.64 |
| **Float32 prox, groups of 5, range, sparse** | ProximalBase | 9.07e+03 ns | 1.19e+04 ns | 1.08e+04 ns | 1.29e+04 ns | 6.31e+03 ns | 8.37e+03 ns |
| | k_direct | 0.71 | 0.55 | 0.65 | 0.52 | 0.75 | 0.60 |
| | po | 0.73 | 0.62 | 0.73 | 0.63 | 0.82 | 0.62 |
| **Float32 prox, groups of 5, perm, allshrink** | ProximalBase | 1.18e+04 ns | 1.12e+04 ns | 1.26e+04 ns | 1.14e+04 ns | 8.98e+03 ns | 8.47e+03 ns |
| | k_direct | 0.76 | 0.77 | 0.79 | 0.77 | 0.79 | 0.79 |
| **Float32 prox, groups of 32, range, allshrink** | ProximalBase | 1.83e+03 ns | 2.13e+03 ns | 1.93e+03 ns | 2.02e+03 ns | 1.26e+03 ns | 1.51e+03 ns |
| | k_direct | 0.80 | 0.64 | 0.78 | 0.77 | 0.78 | 0.73 |
| | po | 1.72 | 1.59 | 2.06 | 2.22 | 1.91 | 1.73 |
| **Float32 prox, groups of 32, range, sparse** | ProximalBase | 1.98e+03 ns | 2.31e+03 ns | 1.79e+03 ns | 2.38e+03 ns | 1.17e+03 ns | 1.48e+03 ns |
| | k_direct | 0.67 | 0.58 | 0.75 | 0.58 | 0.76 | 0.67 |
| | po | 1.52 | 1.43 | 2.20 | 1.89 | 2.07 | 1.76 |
| **Float32 prox, groups of 32, perm, allshrink** | ProximalBase | 6.14e+03 ns | 6.26e+03 ns | 5.95e+03 ns | 6.55e+03 ns | 3.46e+03 ns | 3.89e+03 ns |
| | k_direct | 0.91 | 0.94 | 0.96 | 0.93 | 0.96 | 0.86 |
| **Float32 prox, groups of 1024, range, allshrink** | ProximalBase | 538 ns | 602 ns | 528 ns | 552 ns | 366 ns | 379 ns |
| | k_direct | 0.93 | 0.90 | 1.01 | 0.97 | 0.94 | 0.90 |
| | po | 7.28 | 6.60 | 11.32 | 10.82 | 9.68 | 9.33 |
| **Float32 prox, groups of 1024, range, sparse** | ProximalBase | 589 ns | 585 ns | 562 ns | 530 ns | 353 ns | 385 ns |
| | k_direct | 0.80 | 0.81 | 0.78 | 0.83 | 0.93 | 0.86 |
| | po | 6.70 | 6.77 | 10.61 | 11.25 | 10.01 | 9.21 |
| **Float32 prox, groups of 1024, perm, allshrink** | ProximalBase | 6.83e+03 ns | 6.69e+03 ns | 5.44e+03 ns | 5.47e+03 ns | 3.06e+03 ns | 2.97e+03 ns |
| | k_direct | 0.99 | 1.00 | 1.00 | 0.99 | 0.96 | 0.98 |
| **Float32 valprox, groups of 2, range, allshrink** | ProximalBase | 2.55e+04 ns | 2.92e+04 ns | 2.67e+04 ns | 3.1e+04 ns | 1.7e+04 ns | 2.11e+04 ns |
| | k_direct | 0.68 | 0.60 | 0.80 | 0.65 | 0.69 | 0.47 |
| | po | 0.47 | 0.43 | 0.50 | 0.43 | 0.52 | 0.40 |
| **Float32 valprox, groups of 2, range, sparse** | ProximalBase | 2.04e+04 ns | 2.2e+04 ns | 2.25e+04 ns | 2.57e+04 ns | 1.48e+04 ns | 1.71e+04 ns |
| | k_direct | 0.67 | 0.63 | 0.94 | 0.71 | 0.72 | 0.57 |
| | po | 0.57 | 0.55 | 0.60 | 0.52 | 0.59 | 0.50 |
| **Float32 valprox, groups of 2, perm, allshrink** | ProximalBase | 2.78e+04 ns | 2.86e+04 ns | 2.95e+04 ns | 3.28e+04 ns | 2.02e+04 ns | 1.81e+04 ns |
| | k_direct | 0.67 | 0.71 | 0.67 | 0.65 | 0.53 | 0.62 |
| **Float32 valprox, groups of 5, range, allshrink** | ProximalBase | 1.13e+04 ns | 1.32e+04 ns | 1.2e+04 ns | 1.34e+04 ns | 8.73e+03 ns | 9.62e+03 ns |
| | k_direct | 0.68 | 0.61 | 0.70 | 0.67 | 0.69 | 0.60 |
| | po | 0.60 | 0.56 | 0.66 | 0.61 | 0.59 | 0.54 |
| **Float32 valprox, groups of 5, range, sparse** | ProximalBase | 9.69e+03 ns | 1.21e+04 ns | 1.06e+04 ns | 1.39e+04 ns | 7.1e+03 ns | 9.42e+03 ns |
| | k_direct | 0.66 | 0.53 | 0.75 | 0.51 | 0.69 | 0.48 |
| | po | 0.68 | 0.60 | 0.74 | 0.59 | 0.73 | 0.55 |
| **Float32 valprox, groups of 5, perm, allshrink** | ProximalBase | 1.34e+04 ns | 1.36e+04 ns | 1.59e+04 ns | 1.41e+04 ns | 1.08e+04 ns | 1e+04 ns |
| | k_direct | 0.71 | 0.69 | 0.67 | 0.65 | 0.69 | 0.71 |
| **Float32 valprox, groups of 32, range, allshrink** | ProximalBase | 2.29e+03 ns | 2.93e+03 ns | 2.37e+03 ns | 2.38e+03 ns | 1.61e+03 ns | 1.72e+03 ns |
| | k_direct | 0.68 | 0.50 | 0.67 | 0.68 | 0.67 | 0.69 |
| | po | 1.38 | 1.14 | 1.70 | 1.89 | 1.51 | 1.52 |
| **Float32 valprox, groups of 32, range, sparse** | ProximalBase | 2.08e+03 ns | 2.38e+03 ns | 1.93e+03 ns | 2.52e+03 ns | 1.29e+03 ns | 1.58e+03 ns |
| | k_direct | 0.64 | 0.56 | 0.68 | 0.54 | 0.71 | 0.63 |
| | po | 1.45 | 1.39 | 2.02 | 1.80 | 1.88 | 1.66 |
| **Float32 valprox, groups of 32, perm, allshrink** | ProximalBase | 6.22e+03 ns | 6.53e+03 ns | 6.36e+03 ns | 6.57e+03 ns | 4.25e+03 ns | 4.57e+03 ns |
| | k_direct | 0.91 | 0.92 | 0.91 | 0.93 | 0.76 | 0.77 |
| **Float32 valprox, groups of 1024, range, allshrink** | ProximalBase | 617 ns | 621 ns | 583 ns | 571 ns | 376 ns | 384 ns |
| | k_direct | 0.86 | 0.91 | 0.94 | 0.95 | 0.97 | 0.90 |
| | po | 6.36 | 6.40 | 10.29 | 10.47 | 9.45 | 9.23 |
| **Float32 valprox, groups of 1024, range, sparse** | ProximalBase | 602 ns | 598 ns | 545 ns | 538 ns | 360 ns | 385 ns |
| | k_direct | 0.81 | 0.82 | 0.81 | 0.80 | 0.92 | 0.86 |
| | po | 6.59 | 6.64 | 10.98 | 11.08 | 9.83 | 9.21 |
| **Float32 valprox, groups of 1024, perm, allshrink** | ProximalBase | 6.87e+03 ns | 6.71e+03 ns | 5.5e+03 ns | 5.47e+03 ns | 2.92e+03 ns | 3.04e+03 ns |
| | k_direct | 0.98 | 1.00 | 0.99 | 0.99 | 0.98 | 0.94 |
| **Float32 value, groups of 2, range, allshrink** | ProximalBase | 1.15e+04 ns | 1.34e+04 ns | 1.17e+04 ns | 1.23e+04 ns | 7.76e+03 ns | 8.88e+03 ns |
| | k_direct | 0.52 | 0.49 | 0.65 | 0.56 | 0.51 | 0.47 |
| | po | 0.37 | 0.32 | 0.47 | 0.45 | 0.36 | 0.38 |
| **Float32 value, groups of 2, range, sparse** | ProximalBase | 1.38e+04 ns | 1.45e+04 ns | 1.53e+04 ns | 1.4e+04 ns | 8.94e+03 ns | 1.01e+04 ns |
| | k_direct | 0.43 | 0.44 | 0.50 | 0.49 | 0.44 | 0.41 |
| | po | 0.31 | 0.29 | 0.36 | 0.39 | 0.32 | 0.34 |
| **Float32 value, groups of 2, perm, allshrink** | ProximalBase | 1.19e+04 ns | 1.2e+04 ns | 1.21e+04 ns | 1.25e+04 ns | 8.28e+03 ns | 8.49e+03 ns |
| | k_direct | 0.55 | 0.63 | 0.54 | 0.57 | 0.51 | 0.51 |
| **Float32 value, groups of 5, range, allshrink** | ProximalBase | 5.48e+03 ns | 7.21e+03 ns | 6.33e+03 ns | 6.8e+03 ns | 4.33e+03 ns | 4.75e+03 ns |
| | k_direct | 0.64 | 0.57 | 0.57 | 0.61 | 0.58 | 0.62 |
| | po | 0.62 | 0.37 | 0.52 | 0.53 | 0.47 | 0.43 |
| **Float32 value, groups of 5, range, sparse** | ProximalBase | 7.22e+03 ns | 8.14e+03 ns | 9e+03 ns | 7.66e+03 ns | 5.33e+03 ns | 5.72e+03 ns |
| | k_direct | 0.48 | 0.51 | 0.40 | 0.54 | 0.47 | 0.51 |
| | po | 0.47 | 0.33 | 0.37 | 0.47 | 0.38 | 0.35 |
| **Float32 value, groups of 5, perm, allshrink** | ProximalBase | 6.32e+03 ns | 5.71e+03 ns | 7.35e+03 ns | 5.99e+03 ns | 5.07e+03 ns | 4.39e+03 ns |
| | k_direct | 0.65 | 0.73 | 0.64 | 0.70 | 0.59 | 0.65 |
| **Float32 value, groups of 32, range, allshrink** | ProximalBase | 1.07e+03 ns | 1.14e+03 ns | 1.16e+03 ns | 1.23e+03 ns | 841 ns | 975 ns |
| | k_direct | 0.64 | 0.65 | 0.66 | 0.64 | 0.59 | 0.60 |
| | po | 1.89 | 1.77 | 2.40 | 2.28 | 2.13 | 1.82 |
| **Float32 value, groups of 32, range, sparse** | ProximalBase | 1.32e+03 ns | 1.47e+03 ns | 1.44e+03 ns | 1.58e+03 ns | 1.02e+03 ns | 1.23e+03 ns |
| | k_direct | 0.51 | 0.49 | 0.53 | 0.50 | 0.49 | 0.48 |
| | po | 1.50 | 1.36 | 1.93 | 1.77 | 1.76 | 1.44 |
| **Float32 value, groups of 32, perm, allshrink** | ProximalBase | 2.69e+03 ns | 2.76e+03 ns | 2.54e+03 ns | 2.55e+03 ns | 1.61e+03 ns | 2.17e+03 ns |
| | k_direct | 0.90 | 0.93 | 0.85 | 0.93 | 0.90 | 0.76 |
| **Float32 value, groups of 1024, range, allshrink** | ProximalBase | 203 ns | 253 ns | 216 ns | 213 ns | 155 ns | 158 ns |
| | k_direct | 0.94 | 0.75 | 0.93 | 0.93 | 0.95 | 0.95 |
| | po | 17.33 | 13.93 | 24.78 | 25.07 | 21.09 | 20.69 |
| **Float32 value, groups of 1024, range, sparse** | ProximalBase | 326 ns | 319 ns | 328 ns | 310 ns | 213 ns | 223 ns |
| | k_direct | 0.59 | 0.59 | 0.62 | 0.64 | 0.69 | 0.68 |
| | po | 10.83 | 11.05 | 16.29 | 17.21 | 15.35 | 14.71 |
| **Float32 value, groups of 1024, perm, allshrink** | ProximalBase | 1.58e+03 ns | 1.57e+03 ns | 1.67e+03 ns | 1.68e+03 ns | 1.1e+03 ns | 1.09e+03 ns |
| | k_direct | 0.98 | 1.00 | 1.00 | 0.99 | 0.99 | 0.99 |
| **Float32 block_sweep, groups of 2, range, allshrink** | ProximalBase | 3.47e+04 ns | 3.32e+04 ns | 3.83e+04 ns | 3.61e+04 ns | 2.06e+04 ns | 1.96e+04 ns |
| | k_direct | 0.51 | 0.55 | 0.56 | 0.54 | 0.50 | 0.56 |
| **Float32 block_sweep, groups of 2, range, sparse** | ProximalBase | 3.37e+04 ns | 3.16e+04 ns | 3.74e+04 ns | 3.19e+04 ns | 2.17e+04 ns | 2.09e+04 ns |
| | k_direct | 0.41 | 0.50 | 0.40 | 0.52 | 0.43 | 0.46 |
| **Float32 block_sweep, groups of 5, range, allshrink** | ProximalBase | 1.43e+04 ns | 1.46e+04 ns | 1.62e+04 ns | 1.53e+04 ns | 1.15e+04 ns | 1.04e+04 ns |
| | k_direct | 0.51 | 0.58 | 0.51 | 0.61 | 0.50 | 0.60 |
| **Float32 block_sweep, groups of 5, range, sparse** | ProximalBase | 1.58e+04 ns | 1.48e+04 ns | 1.79e+04 ns | 1.5e+04 ns | 1.01e+04 ns | 1e+04 ns |
| | k_direct | 0.48 | 0.45 | 0.38 | 0.48 | 0.43 | 0.48 |
| **Float32 block_sweep, groups of 32, range, allshrink** | ProximalBase | 2.59e+03 ns | 2.58e+03 ns | 2.86e+03 ns | 2.84e+03 ns | 1.87e+03 ns | 1.88e+03 ns |
| | k_direct | 0.54 | 0.56 | 0.49 | 0.59 | 0.51 | 0.60 |
| **Float32 block_sweep, groups of 32, range, sparse** | ProximalBase | 2.86e+03 ns | 2.94e+03 ns | 3.12e+03 ns | 3.02e+03 ns | 1.84e+03 ns | 1.78e+03 ns |
| | k_direct | 0.45 | 0.48 | 0.42 | 0.43 | 0.50 | 0.55 |
| **Float32 block_sweep, groups of 1024, range, allshrink** | ProximalBase | 598 ns | 606 ns | 575 ns | 586 ns | 363 ns | 362 ns |
| | k_direct | 0.86 | 0.93 | 0.94 | 0.97 | 0.97 | 0.97 |
| **Float32 block_sweep, groups of 1024, range, sparse** | ProximalBase | 595 ns | 600 ns | 584 ns | 539 ns | 393 ns | 359 ns |
| | k_direct | 0.80 | 0.85 | 0.83 | 0.90 | 0.88 | 0.94 |
