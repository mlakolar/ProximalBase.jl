| operation | implementation | Zen3-1.13 | Zen3-1.10 | Broadwell-1.13 | Broadwell-1.10 | SkylakeSP-1.13 | SkylakeSP-1.10 |
|:--|:--|--:|--:|--:|--:|--:|--:|
| **Float64 prox, groups of 2, range, allshrink** | ProximalBase | 1.34e+04 ns | 1.78e+04 ns | 2.23e+04 ns | 2.51e+04 ns | 2.51e+04 ns | 2.58e+04 ns |
| | k_direct | 0.91 | 0.60 | 0.86 | 0.72 | 0.84 | 0.80 |
| | po | 0.76 | 0.56 | 0.60 | 0.57 | 0.60 | 0.58 |
| **Float64 prox, groups of 2, range, sparse** | ProximalBase | 1.38e+04 ns | 1.66e+04 ns | 1.96e+04 ns | 2.16e+04 ns | 2.44e+04 ns | 2.45e+04 ns |
| | k_direct | 0.76 | 0.59 | 0.77 | 0.66 | 0.77 | 0.79 |
| | po | 0.77 | 0.61 | 0.67 | 0.62 | 0.60 | 0.59 |
| **Float64 prox, groups of 2, perm, allshrink** | ProximalBase | 1.39e+04 ns | 1.56e+04 ns | 2.72e+04 ns | 2.72e+04 ns | 2.63e+04 ns | 2.87e+04 ns |
| | k_direct | 0.75 | 0.68 | 0.74 | 0.77 | 0.78 | 0.78 |
| **Float64 prox, groups of 5, range, allshrink** | ProximalBase | 7.99e+03 ns | 8.78e+03 ns | 1.08e+04 ns | 1.19e+04 ns | 1.25e+04 ns | 1.17e+04 ns |
| | k_direct | 0.79 | 0.70 | 0.79 | 0.71 | 0.88 | 0.76 |
| | po | 0.74 | 0.68 | 0.71 | 0.67 | 0.77 | 0.75 |
| **Float64 prox, groups of 5, range, sparse** | ProximalBase | 7.18e+03 ns | 8.07e+03 ns | 1.01e+04 ns | 1.26e+04 ns | 1.11e+04 ns | 1.29e+04 ns |
| | k_direct | 0.81 | 0.66 | 0.76 | 0.61 | 0.91 | 0.58 |
| | po | 0.83 | 0.73 | 0.71 | 0.61 | 0.84 | 0.67 |
| **Float64 prox, groups of 5, perm, allshrink** | ProximalBase | 1.01e+04 ns | 9.25e+03 ns | 1.29e+04 ns | 1.22e+04 ns | 1.38e+04 ns | 1.25e+04 ns |
| | k_direct | 0.81 | 0.80 | 0.80 | 0.85 | 0.82 | 0.80 |
| **Float64 prox, groups of 32, range, allshrink** | ProximalBase | 1.57e+03 ns | 1.82e+03 ns | 2.4e+03 ns | 2.64e+03 ns | 2.65e+03 ns | 2.76e+03 ns |
| | k_direct | 0.84 | 0.76 | 0.83 | 0.76 | 0.81 | 0.78 |
| | po | 1.73 | 1.61 | 1.49 | 1.42 | 1.86 | 1.82 |
| **Float64 prox, groups of 32, range, sparse** | ProximalBase | 1.64e+03 ns | 1.79e+03 ns | 2.3e+03 ns | 2.72e+03 ns | 2.62e+03 ns | 2.89e+03 ns |
| | k_direct | 0.71 | 0.69 | 0.76 | 0.63 | 0.69 | 0.61 |
| | po | 1.66 | 1.67 | 1.49 | 1.31 | 1.85 | 1.71 |
| **Float64 prox, groups of 32, perm, allshrink** | ProximalBase | 4.01e+03 ns | 4.5e+03 ns | 8.75e+03 ns | 8.72e+03 ns | 7.21e+03 ns | 7.89e+03 ns |
| | k_direct | 0.93 | 0.93 | 0.99 | 0.99 | 0.97 | 0.94 |
| **Float64 prox, groups of 1024, range, allshrink** | ProximalBase | 686 ns | 690 ns | 1.45e+03 ns | 1.44e+03 ns | 1.26e+03 ns | 1.24e+03 ns |
| | k_direct | 1.01 | 0.96 | 1.01 | 1.00 | 1.01 | 1.01 |
| | po | 5.41 | 5.37 | 2.88 | 2.89 | 5.04 | 5.11 |
| **Float64 prox, groups of 1024, range, sparse** | ProximalBase | 740 ns | 755 ns | 1.59e+03 ns | 1.58e+03 ns | 1.27e+03 ns | 1.2e+03 ns |
| | k_direct | 0.90 | 0.87 | 0.89 | 0.90 | 0.87 | 0.90 |
| | po | 5.01 | 4.90 | 2.62 | 2.64 | 4.99 | 5.25 |
| **Float64 prox, groups of 1024, perm, allshrink** | ProximalBase | 4.38e+03 ns | 4.43e+03 ns | 9.66e+03 ns | 9.63e+03 ns | 7.54e+03 ns | 7.55e+03 ns |
| | k_direct | 1.01 | 1.00 | 0.99 | 1.00 | 0.99 | 0.99 |
| **Float64 valprox, groups of 2, range, allshrink** | ProximalBase | 1.75e+04 ns | 2.15e+04 ns | 2.82e+04 ns | 3.06e+04 ns | 2.87e+04 ns | 3.14e+04 ns |
| | k_direct | 0.72 | 0.53 | 0.70 | 0.60 | 0.80 | 0.67 |
| | po | 0.58 | 0.47 | 0.48 | 0.46 | 0.53 | 0.48 |
| **Float64 valprox, groups of 2, range, sparse** | ProximalBase | 1.53e+04 ns | 1.74e+04 ns | 2.11e+04 ns | 2.19e+04 ns | 2.41e+04 ns | 2.45e+04 ns |
| | k_direct | 0.75 | 0.58 | 0.70 | 0.65 | 0.84 | 0.74 |
| | po | 0.69 | 0.58 | 0.62 | 0.61 | 0.61 | 0.59 |
| **Float64 valprox, groups of 2, perm, allshrink** | ProximalBase | 2.07e+04 ns | 2.06e+04 ns | 3.05e+04 ns | 3.17e+04 ns | 3.32e+04 ns | 3.5e+04 ns |
| | k_direct | 0.52 | 0.54 | 0.68 | 0.71 | 0.64 | 0.71 |
| **Float64 valprox, groups of 5, range, allshrink** | ProximalBase | 9.76e+03 ns | 1.03e+04 ns | 1.33e+04 ns | 1.4e+04 ns | 1.4e+04 ns | 1.39e+04 ns |
| | k_direct | 0.71 | 0.64 | 0.66 | 0.63 | 0.82 | 0.64 |
| | po | 0.61 | 0.58 | 0.58 | 0.57 | 0.68 | 0.63 |
| **Float64 valprox, groups of 5, range, sparse** | ProximalBase | 7.89e+03 ns | 8.4e+03 ns | 1.08e+04 ns | 1.27e+04 ns | 1.17e+04 ns | 1.3e+04 ns |
| | k_direct | 0.76 | 0.66 | 0.70 | 0.60 | 0.87 | 0.60 |
| | po | 0.76 | 0.71 | 0.66 | 0.60 | 0.79 | 0.66 |
| **Float64 valprox, groups of 5, perm, allshrink** | ProximalBase | 1.21e+04 ns | 1.12e+04 ns | 1.54e+04 ns | 1.5e+04 ns | 1.78e+04 ns | 1.61e+04 ns |
| | k_direct | 0.72 | 0.69 | 0.73 | 0.72 | 0.67 | 0.65 |
| **Float64 valprox, groups of 32, range, allshrink** | ProximalBase | 1.9e+03 ns | 2.14e+03 ns | 2.76e+03 ns | 3e+03 ns | 3.03e+03 ns | 3.15e+03 ns |
| | k_direct | 0.72 | 0.68 | 0.75 | 0.69 | 0.76 | 0.72 |
| | po | 1.44 | 1.37 | 1.29 | 1.25 | 1.63 | 1.59 |
| **Float64 valprox, groups of 32, range, sparse** | ProximalBase | 1.52e+03 ns | 2.11e+03 ns | 2.45e+03 ns | 2.77e+03 ns | 2.77e+03 ns | 2.97e+03 ns |
| | k_direct | 0.76 | 0.59 | 0.70 | 0.63 | 0.64 | 0.60 |
| | po | 1.80 | 1.42 | 1.40 | 1.28 | 1.75 | 1.66 |
| **Float64 valprox, groups of 32, perm, allshrink** | ProximalBase | 4.73e+03 ns | 4.95e+03 ns | 9e+03 ns | 9.05e+03 ns | 7.61e+03 ns | 8.03e+03 ns |
| | k_direct | 0.80 | 0.87 | 0.97 | 0.96 | 0.93 | 0.92 |
| **Float64 valprox, groups of 1024, range, allshrink** | ProximalBase | 709 ns | 694 ns | 1.48e+03 ns | 1.45e+03 ns | 1.28e+03 ns | 1.26e+03 ns |
| | k_direct | 0.98 | 0.97 | 0.99 | 1.00 | 0.99 | 0.98 |
| | po | 5.25 | 5.34 | 2.82 | 2.88 | 4.96 | 5.01 |
| **Float64 valprox, groups of 1024, range, sparse** | ProximalBase | 754 ns | 757 ns | 1.59e+03 ns | 1.58e+03 ns | 1.25e+03 ns | 1.2e+03 ns |
| | k_direct | 0.88 | 0.87 | 0.89 | 0.90 | 0.87 | 0.90 |
| | po | 4.92 | 4.89 | 2.62 | 2.63 | 5.05 | 5.25 |
| **Float64 valprox, groups of 1024, perm, allshrink** | ProximalBase | 4.4e+03 ns | 4.46e+03 ns | 9.66e+03 ns | 9.69e+03 ns | 7.55e+03 ns | 7.57e+03 ns |
| | k_direct | 1.00 | 1.01 | 1.00 | 1.00 | 1.00 | 0.99 |
| **Float64 value, groups of 2, range, allshrink** | ProximalBase | 8.3e+03 ns | 9.45e+03 ns | 1.15e+04 ns | 1.34e+04 ns | 1.24e+04 ns | 1.26e+04 ns |
| | k_direct | 0.62 | 0.57 | 0.57 | 0.49 | 0.67 | 0.54 |
| | po | 0.59 | 0.53 | 0.39 | 0.34 | 0.45 | 0.39 |
| **Float64 value, groups of 2, range, sparse** | ProximalBase | 9.58e+03 ns | 1.13e+04 ns | 1.42e+04 ns | 1.45e+04 ns | 1.49e+04 ns | 1.56e+04 ns |
| | k_direct | 0.54 | 0.46 | 0.45 | 0.44 | 0.55 | 0.42 |
| | po | 0.51 | 0.44 | 0.30 | 0.29 | 0.37 | 0.31 |
| **Float64 value, groups of 2, perm, allshrink** | ProximalBase | 9.54e+03 ns | 9.94e+03 ns | 1.34e+04 ns | 1.37e+04 ns | 1.3e+04 ns | 1.33e+04 ns |
| | k_direct | 0.58 | 0.54 | 0.57 | 0.61 | 0.58 | 0.60 |
| **Float64 value, groups of 5, range, allshrink** | ProximalBase | 4.7e+03 ns | 4.99e+03 ns | 6.24e+03 ns | 7.04e+03 ns | 6.5e+03 ns | 6.95e+03 ns |
| | k_direct | 0.58 | 0.53 | 0.65 | 0.58 | 0.63 | 0.59 |
| | po | 0.45 | 0.43 | 0.46 | 0.48 | 0.55 | 0.46 |
| **Float64 value, groups of 5, range, sparse** | ProximalBase | 5.53e+03 ns | 5.99e+03 ns | 7.66e+03 ns | 8.12e+03 ns | 7.89e+03 ns | 8.23e+03 ns |
| | k_direct | 0.49 | 0.44 | 0.50 | 0.51 | 0.50 | 0.50 |
| | po | 0.39 | 0.36 | 0.37 | 0.42 | 0.46 | 0.38 |
| **Float64 value, groups of 5, perm, allshrink** | ProximalBase | 5.88e+03 ns | 5.09e+03 ns | 7.03e+03 ns | 6.36e+03 ns | 7.98e+03 ns | 6.69e+03 ns |
| | k_direct | 0.60 | 0.60 | 0.69 | 0.77 | 0.64 | 0.69 |
| **Float64 value, groups of 32, range, allshrink** | ProximalBase | 904 ns | 1.1e+03 ns | 1.27e+03 ns | 1.39e+03 ns | 1.41e+03 ns | 1.51e+03 ns |
| | k_direct | 0.70 | 0.63 | 0.72 | 0.68 | 0.70 | 0.65 |
| | po | 2.12 | 1.68 | 1.68 | 1.55 | 2.07 | 1.94 |
| **Float64 value, groups of 32, range, sparse** | ProximalBase | 1.11e+03 ns | 1.49e+03 ns | 2.3e+03 ns | 1.78e+03 ns | 2.53e+03 ns | 1.84e+03 ns |
| | k_direct | 0.56 | 0.46 | 0.40 | 0.51 | 0.38 | 0.53 |
| | po | 1.70 | 1.25 | 0.89 | 1.16 | 1.13 | 1.57 |
| **Float64 value, groups of 32, perm, allshrink** | ProximalBase | 1.78e+03 ns | 2.27e+03 ns | 3.11e+03 ns | 3.21e+03 ns | 2.89e+03 ns | 2.79e+03 ns |
| | k_direct | 0.85 | 0.88 | 0.89 | 0.89 | 0.87 | 0.92 |
| **Float64 value, groups of 1024, range, allshrink** | ProximalBase | 298 ns | 299 ns | 407 ns | 401 ns | 412 ns | 408 ns |
| | k_direct | 0.98 | 0.97 | 0.98 | 0.97 | 0.98 | 0.96 |
| | po | 11.05 | 10.98 | 8.69 | 8.82 | 12.99 | 13.12 |
| **Float64 value, groups of 1024, range, sparse** | ProximalBase | 418 ns | 415 ns | 630 ns | 649 ns | 597 ns | 575 ns |
| | k_direct | 0.70 | 0.70 | 0.62 | 0.61 | 0.66 | 0.69 |
| | po | 7.85 | 7.91 | 5.61 | 5.45 | 8.96 | 9.31 |
| **Float64 value, groups of 1024, perm, allshrink** | ProximalBase | 1.23e+03 ns | 1.23e+03 ns | 1.83e+03 ns | 1.85e+03 ns | 1.7e+03 ns | 1.69e+03 ns |
| | k_direct | 1.01 | 0.99 | 0.99 | 0.99 | 1.00 | 1.00 |
| **Float64 block_sweep, groups of 2, range, allshrink** | ProximalBase | 2.27e+04 ns | 2.17e+04 ns | 3.78e+04 ns | 3.6e+04 ns | 4.14e+04 ns | 3.97e+04 ns |
| | k_direct | 0.47 | 0.47 | 0.55 | 0.57 | 0.50 | 0.52 |
| **Float64 block_sweep, groups of 2, range, sparse** | ProximalBase | 2.29e+04 ns | 2.12e+04 ns | 3.38e+04 ns | 3.21e+04 ns | 3.87e+04 ns | 3.42e+04 ns |
| | k_direct | 0.45 | 0.45 | 0.46 | 0.50 | 0.37 | 0.43 |
| **Float64 block_sweep, groups of 5, range, allshrink** | ProximalBase | 1.19e+04 ns | 1.12e+04 ns | 1.59e+04 ns | 1.59e+04 ns | 1.73e+04 ns | 1.66e+04 ns |
| | k_direct | 0.53 | 0.58 | 0.55 | 0.60 | 0.50 | 0.54 |
| **Float64 block_sweep, groups of 5, range, sparse** | ProximalBase | 1.02e+04 ns | 9.7e+03 ns | 1.66e+04 ns | 1.6e+04 ns | 1.6e+04 ns | 1.64e+04 ns |
| | k_direct | 0.47 | 0.49 | 0.58 | 0.52 | 0.44 | 0.43 |
| **Float64 block_sweep, groups of 32, range, allshrink** | ProximalBase | 2.23e+03 ns | 2.37e+03 ns | 3.39e+03 ns | 3.27e+03 ns | 3.46e+03 ns | 3.63e+03 ns |
| | k_direct | 0.57 | 0.68 | 0.61 | 0.72 | 0.61 | 0.63 |
| **Float64 block_sweep, groups of 32, range, sparse** | ProximalBase | 2.16e+03 ns | 2.17e+03 ns | 3.2e+03 ns | 3.51e+03 ns | 3.44e+03 ns | 3.62e+03 ns |
| | k_direct | 0.56 | 0.64 | 0.55 | 0.62 | 0.51 | 0.52 |
| **Float64 block_sweep, groups of 1024, range, allshrink** | ProximalBase | 692 ns | 704 ns | 1.42e+03 ns | 1.46e+03 ns | 1.13e+03 ns | 1.15e+03 ns |
| | k_direct | 0.99 | 0.99 | 1.00 | 1.05 | 1.01 | 0.99 |
| **Float64 block_sweep, groups of 1024, range, sparse** | ProximalBase | 763 ns | 759 ns | 1.61e+03 ns | 1.61e+03 ns | 1.38e+03 ns | 1.33e+03 ns |
| | k_direct | 0.87 | 0.88 | 0.89 | 0.90 | 0.88 | 0.90 |
| **Float32 prox, groups of 2, range, allshrink** | ProximalBase | 1.1e+04 ns | 1.49e+04 ns | 2.05e+04 ns | 2.4e+04 ns | 2.55e+04 ns | 2.55e+04 ns |
| | k_direct | 0.79 | 0.71 | 0.84 | 0.71 | 0.77 | 0.69 |
| | po | 0.81 | 0.57 | 0.58 | 0.52 | 0.52 | 0.52 |
| **Float32 prox, groups of 2, range, sparse** | ProximalBase | 1.32e+04 ns | 1.59e+04 ns | 1.92e+04 ns | 2.16e+04 ns | 2.42e+04 ns | 2.38e+04 ns |
| | k_direct | 0.74 | 0.63 | 0.72 | 0.66 | 0.76 | 0.70 |
| | po | 0.68 | 0.53 | 0.61 | 0.56 | 0.55 | 0.56 |
| **Float32 prox, groups of 2, perm, allshrink** | ProximalBase | 1.27e+04 ns | 1.37e+04 ns | 2.33e+04 ns | 2.64e+04 ns | 2.43e+04 ns | 2.6e+04 ns |
| | k_direct | 0.65 | 0.69 | 0.76 | 0.82 | 0.77 | 0.75 |
| **Float32 prox, groups of 5, range, allshrink** | ProximalBase | 6.97e+03 ns | 8.08e+03 ns | 9.18e+03 ns | 1.15e+04 ns | 1.02e+04 ns | 1.12e+04 ns |
| | k_direct | 0.80 | 0.74 | 0.80 | 0.69 | 0.77 | 0.70 |
| | po | 0.75 | 0.64 | 0.74 | 0.65 | 0.78 | 0.73 |
| **Float32 prox, groups of 5, range, sparse** | ProximalBase | 6.37e+03 ns | 8.37e+03 ns | 9.08e+03 ns | 1.15e+04 ns | 1.08e+04 ns | 1.29e+04 ns |
| | k_direct | 0.75 | 0.60 | 0.71 | 0.57 | 0.65 | 0.52 |
| | po | 0.82 | 0.62 | 0.73 | 0.63 | 0.73 | 0.63 |
| **Float32 prox, groups of 5, perm, allshrink** | ProximalBase | 8.94e+03 ns | 8.63e+03 ns | 1.19e+04 ns | 1.11e+04 ns | 1.26e+04 ns | 1.14e+04 ns |
| | k_direct | 0.78 | 0.77 | 0.75 | 0.78 | 0.79 | 0.77 |
| **Float32 prox, groups of 32, range, allshrink** | ProximalBase | 1.26e+03 ns | 1.53e+03 ns | 1.83e+03 ns | 2.18e+03 ns | 1.93e+03 ns | 2.02e+03 ns |
| | k_direct | 0.79 | 0.76 | 0.79 | 0.65 | 0.78 | 0.77 |
| | po | 1.92 | 1.75 | 1.73 | 1.54 | 2.06 | 2.22 |
| **Float32 prox, groups of 32, range, sparse** | ProximalBase | 1.23e+03 ns | 1.46e+03 ns | 2.07e+03 ns | 2.35e+03 ns | 1.79e+03 ns | 2.38e+03 ns |
| | k_direct | 0.83 | 0.67 | 0.66 | 0.57 | 0.75 | 0.58 |
| | po | 1.97 | 1.79 | 1.51 | 1.42 | 2.20 | 1.89 |
| **Float32 prox, groups of 32, perm, allshrink** | ProximalBase | 3.6e+03 ns | 3.88e+03 ns | 6.06e+03 ns | 6.3e+03 ns | 5.95e+03 ns | 6.55e+03 ns |
| | k_direct | 0.86 | 0.86 | 0.93 | 0.94 | 0.96 | 0.93 |
| **Float32 prox, groups of 1024, range, allshrink** | ProximalBase | 357 ns | 378 ns | 584 ns | 584 ns | 528 ns | 552 ns |
| | k_direct | 0.95 | 0.90 | 0.86 | 0.82 | 1.01 | 0.97 |
| | po | 9.87 | 9.36 | 6.65 | 6.65 | 11.32 | 10.82 |
| **Float32 prox, groups of 1024, range, sparse** | ProximalBase | 362 ns | 383 ns | 568 ns | 567 ns | 562 ns | 530 ns |
| | k_direct | 0.92 | 0.86 | 0.82 | 0.82 | 0.78 | 0.83 |
| | po | 9.76 | 9.23 | 6.89 | 6.94 | 10.61 | 11.25 |
| **Float32 prox, groups of 1024, perm, allshrink** | ProximalBase | 3.06e+03 ns | 3.06e+03 ns | 6.87e+03 ns | 6.7e+03 ns | 5.44e+03 ns | 5.47e+03 ns |
| | k_direct | 0.96 | 0.96 | 0.98 | 1.00 | 1.00 | 0.99 |
| **Float32 valprox, groups of 2, range, allshrink** | ProximalBase | 1.7e+04 ns | 2.08e+04 ns | 2.57e+04 ns | 2.94e+04 ns | 2.67e+04 ns | 3.1e+04 ns |
| | k_direct | 0.56 | 0.48 | 0.66 | 0.58 | 0.80 | 0.65 |
| | po | 0.53 | 0.41 | 0.46 | 0.42 | 0.50 | 0.43 |
| **Float32 valprox, groups of 2, range, sparse** | ProximalBase | 1.47e+04 ns | 1.72e+04 ns | 2.03e+04 ns | 2.2e+04 ns | 2.25e+04 ns | 2.57e+04 ns |
| | k_direct | 0.70 | 0.57 | 0.67 | 0.63 | 0.94 | 0.71 |
| | po | 0.61 | 0.49 | 0.57 | 0.55 | 0.60 | 0.52 |
| **Float32 valprox, groups of 2, perm, allshrink** | ProximalBase | 1.8e+04 ns | 1.81e+04 ns | 2.79e+04 ns | 2.9e+04 ns | 2.95e+04 ns | 3.28e+04 ns |
| | k_direct | 0.54 | 0.63 | 0.66 | 0.82 | 0.67 | 0.65 |
| **Float32 valprox, groups of 5, range, allshrink** | ProximalBase | 8.74e+03 ns | 9.6e+03 ns | 1.14e+04 ns | 1.32e+04 ns | 1.2e+04 ns | 1.34e+04 ns |
| | k_direct | 0.68 | 0.60 | 0.68 | 0.61 | 0.70 | 0.67 |
| | po | 0.60 | 0.54 | 0.60 | 0.56 | 0.66 | 0.61 |
| **Float32 valprox, groups of 5, range, sparse** | ProximalBase | 7.11e+03 ns | 9.42e+03 ns | 9.83e+03 ns | 1.21e+04 ns | 1.06e+04 ns | 1.39e+04 ns |
| | k_direct | 0.69 | 0.48 | 0.67 | 0.52 | 0.75 | 0.51 |
| | po | 0.74 | 0.55 | 0.68 | 0.60 | 0.74 | 0.59 |
| **Float32 valprox, groups of 5, perm, allshrink** | ProximalBase | 1.08e+04 ns | 1.02e+04 ns | 1.33e+04 ns | 1.36e+04 ns | 1.59e+04 ns | 1.41e+04 ns |
| | k_direct | 0.69 | 0.69 | 0.71 | 0.68 | 0.67 | 0.65 |
| **Float32 valprox, groups of 32, range, allshrink** | ProximalBase | 1.61e+03 ns | 1.73e+03 ns | 2.28e+03 ns | 2.92e+03 ns | 2.37e+03 ns | 2.38e+03 ns |
| | k_direct | 0.66 | 0.69 | 0.68 | 0.52 | 0.67 | 0.68 |
| | po | 1.50 | 1.55 | 1.39 | 1.15 | 1.70 | 1.89 |
| **Float32 valprox, groups of 32, range, sparse** | ProximalBase | 1.33e+03 ns | 1.56e+03 ns | 2.18e+03 ns | 2.37e+03 ns | 1.93e+03 ns | 2.52e+03 ns |
| | k_direct | 0.79 | 0.64 | 0.61 | 0.56 | 0.68 | 0.54 |
| | po | 1.82 | 1.67 | 1.43 | 1.39 | 2.02 | 1.80 |
| **Float32 valprox, groups of 32, perm, allshrink** | ProximalBase | 4.31e+03 ns | 4.39e+03 ns | 6.2e+03 ns | 6.48e+03 ns | 6.36e+03 ns | 6.57e+03 ns |
| | k_direct | 0.74 | 0.80 | 0.92 | 0.93 | 0.91 | 0.93 |
| **Float32 valprox, groups of 1024, range, allshrink** | ProximalBase | 367 ns | 384 ns | 589 ns | 610 ns | 583 ns | 571 ns |
| | k_direct | 0.93 | 0.90 | 0.93 | 0.82 | 0.94 | 0.95 |
| | po | 9.61 | 9.22 | 6.66 | 6.38 | 10.29 | 10.47 |
| **Float32 valprox, groups of 1024, range, sparse** | ProximalBase | 368 ns | 382 ns | 586 ns | 573 ns | 545 ns | 538 ns |
| | k_direct | 0.91 | 0.87 | 0.78 | 0.80 | 0.81 | 0.80 |
| | po | 9.62 | 9.24 | 6.68 | 6.87 | 10.98 | 11.08 |
| **Float32 valprox, groups of 1024, perm, allshrink** | ProximalBase | 3.07e+03 ns | 3e+03 ns | 6.8e+03 ns | 6.77e+03 ns | 5.5e+03 ns | 5.47e+03 ns |
| | k_direct | 0.96 | 0.97 | 1.00 | 0.99 | 0.99 | 0.99 |
| **Float32 value, groups of 2, range, allshrink** | ProximalBase | 7.75e+03 ns | 8.86e+03 ns | 1.15e+04 ns | 1.34e+04 ns | 1.17e+04 ns | 1.23e+04 ns |
| | k_direct | 0.51 | 0.47 | 0.56 | 0.49 | 0.65 | 0.56 |
| | po | 0.36 | 0.38 | 0.37 | 0.32 | 0.47 | 0.45 |
| **Float32 value, groups of 2, range, sparse** | ProximalBase | 8.65e+03 ns | 1e+04 ns | 1.38e+04 ns | 1.45e+04 ns | 1.53e+04 ns | 1.4e+04 ns |
| | k_direct | 0.45 | 0.41 | 0.47 | 0.44 | 0.50 | 0.49 |
| | po | 0.33 | 0.34 | 0.31 | 0.29 | 0.36 | 0.39 |
| **Float32 value, groups of 2, perm, allshrink** | ProximalBase | 8.46e+03 ns | 8.46e+03 ns | 1.18e+04 ns | 1.28e+04 ns | 1.21e+04 ns | 1.25e+04 ns |
| | k_direct | 0.50 | 0.53 | 0.55 | 0.77 | 0.54 | 0.57 |
| **Float32 value, groups of 5, range, allshrink** | ProximalBase | 4.32e+03 ns | 4.73e+03 ns | 5.5e+03 ns | 7.21e+03 ns | 6.33e+03 ns | 6.8e+03 ns |
| | k_direct | 0.57 | 0.62 | 0.65 | 0.57 | 0.57 | 0.61 |
| | po | 0.57 | 0.43 | 0.49 | 0.37 | 0.52 | 0.53 |
| **Float32 value, groups of 5, range, sparse** | ProximalBase | 5.44e+03 ns | 5.72e+03 ns | 7.21e+03 ns | 8.13e+03 ns | 9e+03 ns | 7.66e+03 ns |
| | k_direct | 0.46 | 0.51 | 0.49 | 0.51 | 0.40 | 0.54 |
| | po | 0.45 | 0.35 | 0.37 | 0.33 | 0.37 | 0.47 |
| **Float32 value, groups of 5, perm, allshrink** | ProximalBase | 5.05e+03 ns | 4.57e+03 ns | 6.21e+03 ns | 5.71e+03 ns | 7.35e+03 ns | 5.99e+03 ns |
| | k_direct | 0.59 | 0.65 | 0.65 | 0.73 | 0.64 | 0.70 |
| **Float32 value, groups of 32, range, allshrink** | ProximalBase | 839 ns | 971 ns | 1.06e+03 ns | 1.14e+03 ns | 1.16e+03 ns | 1.23e+03 ns |
| | k_direct | 0.58 | 0.60 | 0.65 | 0.64 | 0.66 | 0.64 |
| | po | 2.10 | 1.82 | 1.90 | 1.76 | 2.40 | 2.28 |
| **Float32 value, groups of 32, range, sparse** | ProximalBase | 1.02e+03 ns | 1.22e+03 ns | 1.33e+03 ns | 1.47e+03 ns | 1.44e+03 ns | 1.58e+03 ns |
| | k_direct | 0.53 | 0.48 | 0.51 | 0.49 | 0.53 | 0.50 |
| | po | 1.73 | 1.45 | 1.50 | 1.36 | 1.93 | 1.77 |
| **Float32 value, groups of 32, perm, allshrink** | ProximalBase | 1.62e+03 ns | 2.14e+03 ns | 2.68e+03 ns | 2.77e+03 ns | 2.54e+03 ns | 2.55e+03 ns |
| | k_direct | 0.87 | 0.77 | 0.90 | 0.93 | 0.85 | 0.93 |
| **Float32 value, groups of 1024, range, allshrink** | ProximalBase | 160 ns | 158 ns | 203 ns | 217 ns | 216 ns | 213 ns |
| | k_direct | 0.95 | 0.95 | 0.95 | 0.91 | 0.93 | 0.93 |
| | po | 20.41 | 20.70 | 17.34 | 16.24 | 24.78 | 25.07 |
| **Float32 value, groups of 1024, range, sparse** | ProximalBase | 214 ns | 222 ns | 309 ns | 311 ns | 328 ns | 310 ns |
| | k_direct | 0.70 | 0.68 | 0.61 | 0.60 | 0.62 | 0.64 |
| | po | 15.30 | 14.72 | 11.40 | 11.35 | 16.29 | 17.21 |
| **Float32 value, groups of 1024, perm, allshrink** | ProximalBase | 1.09e+03 ns | 1.21e+03 ns | 1.56e+03 ns | 1.58e+03 ns | 1.67e+03 ns | 1.68e+03 ns |
| | k_direct | 1.02 | 0.92 | 0.99 | 0.99 | 1.00 | 0.99 |
| **Float32 block_sweep, groups of 2, range, allshrink** | ProximalBase | 2.19e+04 ns | 1.97e+04 ns | 3.51e+04 ns | 3.32e+04 ns | 3.83e+04 ns | 3.61e+04 ns |
| | k_direct | 0.47 | 0.55 | 0.50 | 0.58 | 0.56 | 0.54 |
| **Float32 block_sweep, groups of 2, range, sparse** | ProximalBase | 2.22e+04 ns | 2.1e+04 ns | 3.67e+04 ns | 3.15e+04 ns | 3.74e+04 ns | 3.19e+04 ns |
| | k_direct | 0.41 | 0.45 | 0.42 | 0.52 | 0.40 | 0.52 |
| **Float32 block_sweep, groups of 5, range, allshrink** | ProximalBase | 1.08e+04 ns | 1.05e+04 ns | 1.41e+04 ns | 1.5e+04 ns | 1.62e+04 ns | 1.53e+04 ns |
| | k_direct | 0.54 | 0.60 | 0.52 | 0.60 | 0.51 | 0.61 |
| **Float32 block_sweep, groups of 5, range, sparse** | ProximalBase | 9.7e+03 ns | 1e+04 ns | 1.53e+04 ns | 1.47e+04 ns | 1.79e+04 ns | 1.5e+04 ns |
| | k_direct | 0.45 | 0.48 | 0.49 | 0.46 | 0.38 | 0.48 |
| **Float32 block_sweep, groups of 32, range, allshrink** | ProximalBase | 1.9e+03 ns | 1.89e+03 ns | 2.59e+03 ns | 2.5e+03 ns | 2.86e+03 ns | 2.84e+03 ns |
| | k_direct | 0.51 | 0.60 | 0.53 | 0.59 | 0.49 | 0.59 |
| **Float32 block_sweep, groups of 32, range, sparse** | ProximalBase | 1.81e+03 ns | 1.78e+03 ns | 2.77e+03 ns | 2.89e+03 ns | 3.12e+03 ns | 3.02e+03 ns |
| | k_direct | 0.52 | 0.54 | 0.47 | 0.48 | 0.42 | 0.43 |
| **Float32 block_sweep, groups of 1024, range, allshrink** | ProximalBase | 368 ns | 367 ns | 645 ns | 602 ns | 575 ns | 586 ns |
| | k_direct | 0.97 | 0.99 | 0.84 | 0.92 | 0.94 | 0.97 |
| **Float32 block_sweep, groups of 1024, range, sparse** | ProximalBase | 387 ns | 369 ns | 591 ns | 633 ns | 584 ns | 539 ns |
| | k_direct | 0.88 | 0.98 | 0.82 | 0.82 | 0.83 | 0.90 |
