
## epyc7513a-julia1.13.4 (3 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 2016 of 2016

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3e | control |
|:--|:--|:--|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 0.98 [0.94–1.00] | 0.98 [0.94–1.00] | 0.99 |
| enet | block_loop | medium | 4 | 0.74 [0.68–0.80] | 0.73 [0.68–0.80] | 1.00 |
| enet | block_one | short | 2 | 0.98 [0.95–1.00] | 0.98 [0.95–1.00] | 1.00 |
| enet | prox_alloc | short | 14 | 0.98 [0.85–1.01] | 0.99 [0.87–1.01] | 1.00 |
| enet | prox_alloc | medium | 14 | 0.93 [0.90–1.06] | 0.96 [0.89–1.01] | 1.01 |
| enet | prox_alloc | large | 12 | 0.98 [0.90–1.07] | 1.00 [0.83–1.22] | 1.02 |
| enet | prox_g_out | short | 14 | 1.00 [0.96–1.08] | 1.01 [1.00–1.04] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [0.99–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | short | 14 | 0.86 [0.72–1.00] | 0.88 [0.72–1.05] | 1.00 |
| enet | prox_in | medium | 14 | 0.70 [0.69–0.71] | 0.70 [0.69–0.71] | 1.00 |
| enet | prox_in | large | 12 | 0.69 [0.69–0.69] | 0.69 [0.69–0.69] | 1.00 |
| enet | prox_incopy | short | 14 | 0.89 [0.74–1.00] | 0.89 [0.75–1.02] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 |
| enet | prox_incopy | large | 12 | 0.73 [0.72–0.75] | 0.73 [0.72–0.75] | 1.00 |
| enet | prox_out | short | 14 | 0.95 [0.74–1.00] | 0.97 [0.74–1.03] | 1.01 |
| enet | prox_out | medium | 14 | 0.71 [0.68–0.81] | 0.77 [0.68–1.00] | 1.00 |
| enet | prox_out | large | 12 | 0.90 [0.69–1.32] | 1.00 [0.99–1.00] | 1.00 |
| enet | sparse_prox | large | 2 | 1.09 [1.09–1.09] | 1.10 [1.09–1.11] | 0.94 |
| enet | valprox_out | short | 14 | 0.98 [0.88–1.02] | 0.99 [0.90–1.03] | 1.00 |
| enet | valprox_out | medium | 14 | 0.79 [0.77–0.89] | 0.83 [0.76–1.01] | 1.00 |
| enet | valprox_out | large | 12 | 0.94 [0.78–1.23] | 1.00 [0.99–1.01] | 1.00 |
| enetw | block_loop | short | 4 | 1.01 [0.91–1.11] | 1.00 [0.90–1.11] | 1.00 |
| enetw | block_loop | medium | 4 | 0.95 [0.86–1.02] | 0.88 [0.86–0.90] | 1.00 |
| enetw | block_one | short | 2 | 0.98 [0.97–1.00] | 0.98 [0.97–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.99 [0.97–1.01] | 0.99 [0.94–1.02] | 1.00 |
| enetw | prox_alloc | medium | 14 | 0.98 [0.90–1.06] | 0.98 [0.90–1.04] | 1.01 |
| enetw | prox_alloc | large | 12 | 1.02 [0.91–1.08] | 0.99 [0.85–1.12] | 1.01 |
| enetw | prox_g_out | short | 14 | 1.00 [0.98–1.01] | 1.00 [0.97–1.02] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [0.99–1.00] | 1.00 [0.99–1.00] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [0.99–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | short | 14 | 0.92 [0.82–1.02] | 0.92 [0.82–1.02] | 1.00 |
| enetw | prox_in | medium | 14 | 0.78 [0.78–0.80] | 0.78 [0.78–0.80] | 1.00 |
| enetw | prox_in | large | 12 | 0.78 [0.78–0.83] | 0.78 [0.78–0.83] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.92 [0.83–1.02] | 0.93 [0.83–1.02] | 1.01 |
| enetw | prox_incopy | medium | 14 | 0.80 [0.79–0.81] | 0.80 [0.79–0.81] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.81 [0.79–0.85] | 0.81 [0.80–0.84] | 1.00 |
| enetw | prox_out | short | 14 | 0.98 [0.88–1.02] | 0.98 [0.87–1.02] | 1.00 |
| enetw | prox_out | medium | 14 | 0.84 [0.73–0.98] | 0.84 [0.73–1.01] | 1.00 |
| enetw | prox_out | large | 12 | 1.02 [0.96–1.16] | 1.00 [1.00–1.01] | 1.00 |
| enetw | sparse_prox | large | 2 | 1.07 [1.06–1.08] | 1.03 [1.00–1.06] | 0.99 |
| enetw | valprox_out | short | 14 | 0.97 [0.83–1.00] | 0.97 [0.84–1.02] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.90 [0.84–0.99] | 0.90 [0.84–1.01] | 1.00 |
| enetw | valprox_out | large | 12 | 1.01 [0.98–1.06] | 1.00 [1.00–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 0.98 [0.96–1.00] | 0.98 [0.96–1.00] | 1.00 |
| l1 | block_loop | medium | 4 | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 1.00 |
| l1 | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.99 [0.97–1.01] | 1.02 [0.94–1.05] | 0.98 |
| l1 | prox_alloc | medium | 14 | 0.96 [0.92–1.02] | 0.96 [0.92–1.02] | 1.00 |
| l1 | prox_alloc | large | 12 | 1.06 [0.91–1.29] | 0.99 [0.82–1.19] | 1.00 |
| l1 | prox_g_out | short | 14 | 1.00 [0.93–1.05] | 1.01 [0.93–1.05] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.99 [0.98–1.01] | 0.99 [0.98–1.00] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.98 [0.97–0.99] | 0.98 [0.96–0.99] | 1.00 |
| l1 | prox_in | short | 14 | 0.84 [0.68–1.00] | 0.84 [0.68–1.00] | 1.00 |
| l1 | prox_in | medium | 14 | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 1.00 |
| l1 | prox_in | large | 12 | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.88 [0.71–1.03] | 0.90 [0.71–1.09] | 1.01 |
| l1 | prox_incopy | medium | 14 | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 |
| l1 | prox_out | short | 14 | 0.96 [0.77–1.05] | 1.01 [0.79–1.14] | 1.00 |
| l1 | prox_out | medium | 14 | 0.68 [0.67–0.79] | 0.74 [0.67–1.00] | 1.00 |
| l1 | prox_out | large | 12 | 0.81 [0.67–1.01] | 0.99 [0.92–1.00] | 0.99 |
| l1 | sparse_prox | large | 2 | 1.02 [1.01–1.02] | 1.02 [1.02–1.02] | 1.00 |
| l1 | valprox_out | short | 14 | 0.96 [0.86–1.00] | 0.98 [0.86–1.05] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.77 [0.74–0.86] | 0.82 [0.74–1.00] | 1.00 |
| l1 | valprox_out | large | 12 | 0.86 [0.75–1.02] | 0.99 [0.98–1.00] | 1.00 |
| l1w | block_loop | short | 4 | 1.01 [0.96–1.05] | 0.99 [0.93–1.05] | 0.99 |
| l1w | block_loop | medium | 4 | 0.73 [0.69–0.78] | 0.72 [0.68–0.78] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.04 |
| l1w | prox_alloc | short | 14 | 0.98 [0.92–1.00] | 1.00 [0.94–1.03] | 1.01 |
| l1w | prox_alloc | medium | 14 | 0.99 [0.89–1.13] | 0.97 [0.90–1.06] | 1.00 |
| l1w | prox_alloc | large | 12 | 1.07 [0.99–1.11] | 0.97 [0.81–1.04] | 0.99 |
| l1w | prox_g_out | short | 14 | 0.99 [0.90–1.04] | 1.01 [0.91–1.10] | 0.97 |
| l1w | prox_g_out | medium | 14 | 0.98 [0.95–1.02] | 1.03 [0.96–1.11] | 0.96 |
| l1w | prox_g_out | large | 12 | 0.98 [0.96–1.00] | 1.03 [0.95–1.11] | 0.96 |
| l1w | prox_in | short | 14 | 0.89 [0.77–1.00] | 0.89 [0.75–1.00] | 1.01 |
| l1w | prox_in | medium | 14 | 0.72 [0.71–0.73] | 0.71 [0.71–0.73] | 1.00 |
| l1w | prox_in | large | 12 | 0.71 [0.71–0.71] | 0.71 [0.71–0.71] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.91 [0.78–1.00] | 0.94 [0.78–1.12] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.74 [0.72–0.76] | 0.74 [0.72–0.76] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 1.00 |
| l1w | prox_out | short | 14 | 0.96 [0.82–1.00] | 1.02 [0.86–1.09] | 1.02 |
| l1w | prox_out | medium | 14 | 0.79 [0.66–1.00] | 0.79 [0.66–1.00] | 1.00 |
| l1w | prox_out | large | 12 | 1.01 [0.99–1.08] | 1.00 [0.99–1.01] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.01 [1.01–1.01] | 1.01 [1.01–1.02] | 1.00 |
| l1w | valprox_out | short | 14 | 0.96 [0.89–1.00] | 0.99 [0.91–1.03] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.89 [0.83–1.00] | 0.89 [0.83–1.01] | 1.00 |
| l1w | valprox_out | large | 12 | 1.01 [0.99–1.05] | 1.00 [0.99–1.00] | 1.00 |

A3M4: 10 of 1008 cells slower in every process, 384 faster in every process, 27 above 1.05 in some processes only, 51 undecided (control unclean)
  slower  enet/prox_out/Float64/n65536/normal          1.294–1.340   14102.4 -> 18602.6 ns
  slower  enet/prox_out/Float32/n1048576/normal        1.267–1.326   111598.4 -> 145002.3 ns
  slower  enet/valprox_out/Float64/n65536/normal       1.212–1.249   19668.8 -> 24294.2 ns
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.111   9.8 -> 10.9 ns
  slower  l1w/prox_alloc/Float64/n8192/normal          1.099–1.145   2607.5 -> 2940.9 ns
  slower  enetw/prox_out/Float32/n262144/normal        1.091–1.095   36016.9 -> 39297.4 ns
  slower  l1/prox_alloc/Float64/n100000/normal         1.080–1.107   28686.5 -> 31895.0 ns
  slower  l1w/prox_alloc/Float64/n32768/normal         1.072–1.118   10947.5 -> 12016.4 ns
  slower  enetw/prox_out/Float32/n1048576/normal       1.070–1.082   139751.7 -> 149664.8 ns
  slower  enetw/prox_out/Float64/n65536/normal         1.069–1.097   17387.3 -> 18755.5 ns
  some    l1/prox_incopy/Float64/n4/normal             0.943–1.954   9.6 -> 9.3 ns   output offsets [128, 64, 64]
  some    enet/prox_out/Float64/n32768/normal          0.702–1.350   7132.3 -> 9352.9 ns   output offsets [64, 1920, 64]
  some    l1/prox_g_out/Float64/n32/normal             0.979–1.334   42.6 -> 41.7 ns   output offsets [288, 288, 288]
  some    enet/valprox_out/Float64/n32768/normal       0.778–1.246   9485.1 -> 11778.6 ns   output offsets [64, 1920, 64]
  some    enet/valprox_out/Float64/n8192/normal        0.778–1.237   2336.5 -> 1817.4 ns   output offsets [2752, 128, 2112]
  some    enetw/valprox_out/Float64/n8/normal          0.966–1.199   22.7 -> 22.5 ns   output offsets [96, 96, 96]
  some    enetw/prox_out/Float64/n8192/normal          0.979–1.197   1741.1 -> 1707.6 ns   output offsets [2752, 128, 2112]
  some    l1/prox_out/Float64/n8192/normal             0.672–1.195   1690.3 -> 1137.4 ns   output offsets [2752, 128, 2112]
  some    enetw/valprox_out/Float64/n4/normal          0.987–1.183   21.2 -> 21.1 ns   output offsets [128, 64, 64]
  some    l1w/prox_alloc/Float64/n1000/normal          0.994–1.172   487.4 -> 518.6 ns   output offsets [2688, 3712, 3072]
  some    enetw/prox_out/Float64/n32768/normal         1.000–1.162   7981.0 -> 9220.5 ns   output offsets [64, 1920, 64]
  some    l1/valprox_out/Float64/n8192/normal          0.754–1.152   2294.0 -> 1740.2 ns   output offsets [2752, 128, 2112]
  some    enetw/valprox_out/Float64/n16/normal         0.953–1.144   25.0 -> 24.0 ns   output offsets [160, 160, 160]
  some    l1w/prox_alloc/Float64/n100000/normal        1.039–1.106   33635.2 -> 36307.6 ns   output offsets [1344, 1344, 1344]
  some    l1w/prox_out/Float64/n32768/normal           0.999–1.097   8303.3 -> 8970.7 ns   output offsets [64, 1920, 64]
  some    enetw/valprox_out/Float64/n32768/normal      0.999–1.074   13390.3 -> 14340.9 ns   output offsets [64, 1920, 64]
  some    enetw/valprox_out/Float32/n262144/normal     1.039–1.059   59105.1 -> 61402.2 ns   output offsets [64, 64, 64]
  some    enet/prox_g_out/Float32/n32/normal           1.019–1.058   25.8 -> 27.1 ns   output offsets [160, 160, 160]
  some    enetw/valprox_out/Float64/n8192/normal       0.989–1.057   2873.0 -> 2850.1 ns   output offsets [2752, 128, 2112]
  some    l1/prox_out/Float32/n1/normal                0.999–1.056   4.9 -> 4.9 ns   output offsets [64, 64, 64]
  some    l1/prox_in/Float32/n1/normal                 1.000–1.056   4.9 -> 4.9 ns   output offsets [64, 64, 64]
  some    enetw/valprox_out/Float64/n65536/normal      1.018–1.055   28742.4 -> 29960.9 ns   output offsets [64, 64, 64]
  some    l1w/valprox_out/Float64/n32768/normal        1.000–1.054   14080.5 -> 14760.5 ns   output offsets [64, 1920, 64]
  some    enetw/valprox_out/Float32/n1048576/normal    1.041–1.050   230068.7 -> 240440.1 ns   output offsets [64, 64, 64]
  some    l1/prox_in/Float32/n2/normal                 0.950–1.050   5.5 -> 5.5 ns   output offsets [64, 64, 192]

A3e: 5 of 1008 cells slower in every process, 344 faster in every process, 14 above 1.05 in some processes only, 47 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 10.9 ns
  slower  l1/prox_out/Float32/n1/normal                1.055–1.056   4.9 -> 5.2 ns
  slower  l1w/prox_out/Float64/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_out/Float32/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1/prox_out/Float32/n4/normal                1.050–1.100   5.5 -> 6.0 ns
  some    l1/prox_incopy/Float64/n4/normal             0.971–1.562   9.6 -> 9.3 ns   output offsets [128, 64, 64]
  some    enetw/valprox_out/Float64/n4/normal          0.974–1.501   21.2 -> 21.3 ns   output offsets [128, 64, 64]
  some    enetw/valprox_out/Float64/n8/normal          0.985–1.421   22.7 -> 23.0 ns   output offsets [96, 96, 96]
  some    l1/prox_g_out/Float64/n32/normal             0.979–1.347   42.6 -> 41.7 ns   output offsets [288, 288, 288]
  some    l1w/valprox_out/Float64/n8/normal            1.002–1.070   13.4 -> 13.7 ns   output offsets [96, 96, 96]
  some    l1/prox_alloc/Float32/n4/normal              1.029–1.062   18.2 -> 19.0 ns   output offsets [48, 48, 96]
  some    l1/prox_out/Float64/n1/normal                1.000–1.056   4.9 -> 4.9 ns   output offsets [64, 64, 64]
  some    l1/prox_in/Float32/n1/normal                 1.000–1.056   4.9 -> 4.9 ns   output offsets [64, 64, 64]
  some    l1/prox_alloc/Float32/n8/normal              1.014–1.054   19.1 -> 19.9 ns   output offsets [64, 64, 64]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.2 ns   output offsets [128, 64, 64]
  some    l1/prox_alloc/Float32/n16/normal             0.995–1.051   20.2 -> 21.1 ns   output offsets [96, 96, 96]
  some    l1/prox_out/Float32/n2/normal                1.050–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 192]
  some    l1/prox_out/Float64/n4/normal                1.000–1.050   5.5 -> 5.7 ns   output offsets [128, 64, 64]
  some    l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 64]

## epyc7513b-julia1.13.4 (3 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 2016 of 2016

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3e | control |
|:--|:--|:--|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 0.98 [0.96–1.01] | 0.98 [0.95–1.00] | 1.00 |
| enet | block_loop | medium | 4 | 0.74 [0.68–0.80] | 0.73 [0.69–0.80] | 1.00 |
| enet | block_one | short | 2 | 0.98 [0.95–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_alloc | short | 14 | 0.98 [0.91–1.00] | 0.99 [0.88–1.01] | 0.99 |
| enet | prox_alloc | medium | 14 | 0.94 [0.91–0.98] | 0.95 [0.86–1.02] | 0.99 |
| enet | prox_alloc | large | 12 | 0.99 [0.86–1.11] | 1.01 [0.71–1.23] | 0.98 |
| enet | prox_g_out | short | 14 | 1.00 [0.99–1.02] | 1.01 [0.98–1.03] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.01] | 1.00 [0.99–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | short | 14 | 0.86 [0.72–1.00] | 0.87 [0.71–1.00] | 1.00 |
| enet | prox_in | medium | 14 | 0.70 [0.69–0.71] | 0.70 [0.69–0.71] | 1.00 |
| enet | prox_in | large | 12 | 0.69 [0.69–0.69] | 0.69 [0.69–0.69] | 1.00 |
| enet | prox_incopy | short | 14 | 0.89 [0.74–1.00] | 0.89 [0.73–1.02] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 |
| enet | prox_incopy | large | 12 | 0.73 [0.72–0.75] | 0.73 [0.72–0.75] | 1.00 |
| enet | prox_out | short | 14 | 0.94 [0.73–1.00] | 0.97 [0.73–1.03] | 1.00 |
| enet | prox_out | medium | 14 | 0.71 [0.68–0.78] | 0.77 [0.68–1.00] | 1.00 |
| enet | prox_out | large | 12 | 0.87 [0.70–1.33] | 1.00 [1.00–1.00] | 1.00 |
| enet | sparse_prox | large | 2 | 1.09 [1.08–1.09] | 1.09 [1.08–1.09] | 0.99 |
| enet | valprox_out | short | 14 | 0.98 [0.88–1.01] | 0.99 [0.89–1.02] | 1.00 |
| enet | valprox_out | medium | 14 | 0.79 [0.76–0.86] | 0.84 [0.76–1.00] | 1.00 |
| enet | valprox_out | large | 12 | 0.90 [0.78–1.21] | 1.00 [0.99–1.00] | 1.00 |
| enetw | block_loop | short | 4 | 1.01 [0.90–1.14] | 1.00 [0.90–1.11] | 1.00 |
| enetw | block_loop | medium | 4 | 0.89 [0.86–0.95] | 0.90 [0.86–0.95] | 1.00 |
| enetw | block_one | short | 2 | 0.98 [0.97–1.00] | 0.98 [0.97–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.99 [0.94–1.01] | 0.99 [0.93–1.06] | 1.01 |
| enetw | prox_alloc | medium | 14 | 0.97 [0.91–1.04] | 0.97 [0.91–1.01] | 0.99 |
| enetw | prox_alloc | large | 12 | 1.02 [0.94–1.07] | 1.01 [0.76–1.24] | 0.97 |
| enetw | prox_g_out | short | 14 | 0.99 [0.98–1.00] | 1.00 [0.98–1.01] | 0.99 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | short | 14 | 0.91 [0.80–1.02] | 0.92 [0.81–1.05] | 1.01 |
| enetw | prox_in | medium | 14 | 0.78 [0.78–0.80] | 0.78 [0.78–0.80] | 1.00 |
| enetw | prox_in | large | 12 | 0.78 [0.78–0.83] | 0.78 [0.78–0.82] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.92 [0.82–1.02] | 0.92 [0.82–1.04] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.80 [0.79–0.82] | 0.80 [0.79–0.82] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.81 [0.79–0.84] | 0.81 [0.79–0.84] | 1.00 |
| enetw | prox_out | short | 14 | 0.97 [0.86–1.02] | 0.99 [0.88–1.05] | 1.00 |
| enetw | prox_out | medium | 14 | 0.84 [0.74–0.98] | 0.85 [0.73–1.00] | 1.00 |
| enetw | prox_out | large | 12 | 1.02 [0.96–1.18] | 1.00 [1.00–1.00] | 1.00 |
| enetw | sparse_prox | large | 2 | 1.07 [1.06–1.07] | 1.04 [1.00–1.08] | 1.00 |
| enetw | valprox_out | short | 14 | 0.96 [0.85–1.01] | 0.98 [0.85–1.03] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.90 [0.84–0.99] | 0.91 [0.83–1.00] | 1.00 |
| enetw | valprox_out | large | 12 | 1.02 [0.98–1.11] | 1.00 [1.00–1.01] | 1.00 |
| l1 | block_loop | short | 4 | 0.97 [0.92–1.00] | 0.97 [0.92–1.00] | 1.00 |
| l1 | block_loop | medium | 4 | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 1.00 |
| l1 | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.99 [0.96–1.01] | 1.02 [0.98–1.04] | 0.99 |
| l1 | prox_alloc | medium | 14 | 0.95 [0.84–1.02] | 0.94 [0.75–1.01] | 1.00 |
| l1 | prox_alloc | large | 12 | 1.00 [0.81–1.26] | 1.04 [0.74–1.26] | 0.96 |
| l1 | prox_g_out | short | 14 | 1.00 [0.92–1.05] | 1.00 [0.92–1.05] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.98 [0.97–0.99] | 0.98 [0.97–0.99] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.98 [0.97–0.99] | 0.98 [0.97–0.98] | 1.00 |
| l1 | prox_in | short | 14 | 0.83 [0.68–1.00] | 0.83 [0.68–1.00] | 1.00 |
| l1 | prox_in | medium | 14 | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 1.00 |
| l1 | prox_in | large | 12 | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.87 [0.70–1.00] | 0.89 [0.71–1.06] | 0.99 |
| l1 | prox_incopy | medium | 14 | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 |
| l1 | prox_out | short | 14 | 0.96 [0.77–1.05] | 0.99 [0.79–1.10] | 1.00 |
| l1 | prox_out | medium | 14 | 0.68 [0.67–0.77] | 0.74 [0.67–1.00] | 1.00 |
| l1 | prox_out | large | 12 | 0.78 [0.67–1.01] | 1.00 [0.96–1.01] | 1.00 |
| l1 | sparse_prox | large | 2 | 1.02 [1.02–1.02] | 1.00 [0.99–1.01] | 0.98 |
| l1 | valprox_out | short | 14 | 0.97 [0.85–1.00] | 0.97 [0.86–1.03] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.77 [0.74–0.86] | 0.82 [0.74–1.00] | 1.00 |
| l1 | valprox_out | large | 12 | 0.85 [0.76–1.05] | 1.00 [0.98–1.00] | 1.00 |
| l1w | block_loop | short | 4 | 0.97 [0.93–1.00] | 0.97 [0.90–1.05] | 0.99 |
| l1w | block_loop | medium | 4 | 0.72 [0.69–0.76] | 0.72 [0.68–0.78] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 0.98 [0.92–1.00] | 1.01 [0.94–1.11] | 1.00 |
| l1w | prox_alloc | medium | 14 | 0.98 [0.90–1.09] | 0.96 [0.89–1.04] | 0.98 |
| l1w | prox_alloc | large | 12 | 1.06 [0.98–1.11] | 0.98 [0.77–1.19] | 0.97 |
| l1w | prox_g_out | short | 14 | 1.02 [0.91–1.10] | 1.02 [0.91–1.10] | 1.00 |
| l1w | prox_g_out | medium | 14 | 1.03 [0.96–1.11] | 1.03 [0.96–1.11] | 1.00 |
| l1w | prox_g_out | large | 12 | 1.03 [0.96–1.11] | 1.02 [0.96–1.10] | 1.00 |
| l1w | prox_in | short | 14 | 0.90 [0.76–1.00] | 0.90 [0.76–1.00] | 1.00 |
| l1w | prox_in | medium | 14 | 0.72 [0.71–0.73] | 0.72 [0.71–0.73] | 1.00 |
| l1w | prox_in | large | 12 | 0.71 [0.71–0.71] | 0.71 [0.71–0.71] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.91 [0.77–1.04] | 0.94 [0.78–1.12] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.74 [0.72–0.76] | 0.74 [0.72–0.76] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 1.00 |
| l1w | prox_out | short | 14 | 0.95 [0.83–1.00] | 1.00 [0.85–1.08] | 1.00 |
| l1w | prox_out | medium | 14 | 0.79 [0.65–0.99] | 0.80 [0.67–1.00] | 1.00 |
| l1w | prox_out | large | 12 | 1.00 [0.99–1.03] | 1.00 [0.99–1.00] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.01 [1.01–1.02] | 1.02 [1.01–1.02] | 1.00 |
| l1w | valprox_out | short | 14 | 0.96 [0.91–1.00] | 0.99 [0.92–1.06] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.89 [0.83–1.00] | 0.89 [0.83–1.00] | 1.00 |
| l1w | valprox_out | large | 12 | 1.00 [0.99–1.02] | 1.00 [0.99–1.00] | 1.00 |

A3M4: 11 of 1008 cells slower in every process, 394 faster in every process, 26 above 1.05 in some processes only, 43 undecided (control unclean)
  slower  enet/prox_out/Float32/n1048576/normal        1.290–1.410   113740.4 -> 148376.1 ns
  slower  enet/prox_out/Float32/n262144/normal         1.231–1.411   28567.6 -> 36736.0 ns
  slower  enet/valprox_out/Float32/n1048576/normal     1.210–1.308   158777.7 -> 192795.5 ns
  slower  enet/valprox_out/Float32/n262144/normal      1.174–1.303   39783.6 -> 48669.3 ns
  slower  enet/valprox_out/Float64/n65536/normal       1.167–1.200   20072.2 -> 24083.7 ns
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 11.2 ns
  slower  enetw/prox_out/Float32/n262144/normal        1.090–1.300   35044.1 -> 40886.3 ns
  slower  enet/sparse_prox/Float32/n65536/tenth        1.082–1.182   25997.7 -> 28439.3 ns
  slower  enetw/prox_out/Float32/n1048576/normal       1.081–1.298   140390.4 -> 154023.4 ns
  slower  enetw/valprox_out/Float32/n1048576/normal    1.056–1.151   230525.6 -> 254611.1 ns
  slower  enetw/valprox_out/Float32/n262144/normal     1.056–1.181   57568.2 -> 63387.3 ns
  some    enet/prox_out/Float64/n65536/normal          0.824–1.332   14009.4 -> 17296.3 ns   output offsets [64, 64, 128]
  some    enet/prox_out/Float32/n32768/normal          0.698–1.302   3380.4 -> 2370.9 ns   output offsets [3136, 64, 2496]
  some    enet/prox_out/Float64/n32768/normal          0.693–1.265   6890.6 -> 5123.0 ns   output offsets [1792, 1920, 64]
  some    l1/prox_out/Float32/n1048576/normal          0.925–1.228   118302.5 -> 120527.1 ns   output offsets [64, 64, 128]
  some    enet/valprox_out/Float32/n32768/normal       0.778–1.227   4609.8 -> 3604.3 ns   output offsets [3136, 64, 2496]
  some    enet/valprox_out/Float64/n32768/normal       0.775–1.194   9704.0 -> 7835.9 ns   output offsets [1792, 1920, 64]
  some    enetw/prox_out/Float32/n32768/normal         0.957–1.179   3543.3 -> 3426.3 ns   output offsets [3136, 64, 2496]
  some    l1/valprox_out/Float32/n262144/normal        0.896–1.165   44259.4 -> 42188.9 ns   output offsets [64, 64, 128]
  some    enet/prox_alloc/Float64/n1/normal            0.999–1.129   23.0 -> 23.1 ns   output offsets [64, 64, 64]
  some    enetw/valprox_out/Float32/n32768/normal      0.975–1.118   5790.8 -> 5709.6 ns   output offsets [3136, 64, 2496]
  some    enet/prox_alloc/Float64/n2/normal            0.999–1.117   23.7 -> 23.8 ns   output offsets [48, 48, 48]
  some    l1w/prox_out/Float32/n32768/normal           0.986–1.116   3430.0 -> 3394.1 ns   output offsets [3136, 64, 2496]
  some    l1/valprox_out/Float32/n32768/normal         0.754–1.096   4535.0 -> 3426.0 ns   output offsets [3136, 64, 2496]
  some    l1/prox_out/Float32/n8/normal                1.045–1.095   5.7 -> 6.3 ns   output offsets [64, 256, 192]
  some    l1w/prox_out/Float32/n262144/normal          0.995–1.085   35298.5 -> 37182.5 ns   output offsets [64, 64, 128]
  some    enetw/prox_out/Float64/n32768/normal         0.997–1.083   8000.5 -> 8009.6 ns   output offsets [1792, 1920, 64]
  some    l1/prox_out/Float32/n16/normal               0.960–1.080   6.8 -> 7.4 ns   output offsets [96, 96, 96]
  some    l1w/valprox_out/Float32/n32768/normal        0.997–1.075   6248.3 -> 6230.4 ns   output offsets [3136, 64, 2496]
  some    l1w/prox_out/Float32/n1048576/normal         1.023–1.073   141735.7 -> 145452.0 ns   output offsets [64, 64, 128]
  some    l1w/prox_alloc/Float64/n100000/normal        0.934–1.060   34271.7 -> 36357.1 ns   output offsets [1344, 1344, 1408]
  some    l1/prox_out/Float32/n1/normal                1.000–1.056   4.9 -> 4.9 ns   output offsets [64, 96, 64]
  some    l1/prox_in/Float32/n1/normal                 1.000–1.056   4.9 -> 4.9 ns   output offsets [64, 96, 64]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.2 ns   output offsets [64, 64, 128]
  some    l1w/valprox_out/Float32/n1048576/normal      1.014–1.052   232534.9 -> 237916.0 ns   output offsets [64, 64, 128]
  some    l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns   output offsets [64, 96, 64]

A3e: 5 of 1008 cells slower in every process, 348 faster in every process, 13 above 1.05 in some processes only, 42 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 10.9 ns
  slower  l1/prox_out/Float32/n1/normal                1.056–1.056   4.9 -> 5.2 ns
  slower  l1w/prox_out/Float64/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_out/Float32/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1/prox_out/Float32/n4/normal                1.050–1.100   5.5 -> 5.7 ns
  some    enet/sparse_prox/Float32/n65536/tenth        0.947–1.169   25997.7 -> 28123.2 ns   output offsets [2432, 1664, 448]
  some    enetw/valprox_out/Float64/n32/normal         0.956–1.164   28.4 -> 27.3 ns   output offsets [288, 288, 288]
  some    l1/prox_out/Float32/n8/normal                1.045–1.143   5.7 -> 6.3 ns   output offsets [64, 256, 192]
  some    l1/prox_out/Float32/n16/normal               0.962–1.120   6.8 -> 6.8 ns   output offsets [96, 96, 96]
  some    enet/valprox_out/Float64/n1/normal           1.000–1.063   15.6 -> 15.8 ns   output offsets [64, 64, 64]
  some    l1/prox_alloc/Float32/n1/normal              1.031–1.062   17.9 -> 18.2 ns   output offsets [64, 96, 64]
  some    enetw/prox_out/Float32/n32768/normal         1.000–1.062   3543.3 -> 3543.3 ns   output offsets [3136, 64, 2496]
  some    l1/prox_out/Float32/n1048576/normal          0.912–1.058   118302.5 -> 118890.2 ns   output offsets [64, 64, 128]
  some    enetw/prox_out/Float64/n16384/normal         0.999–1.056   3494.6 -> 3491.2 ns   output offsets [64, 448, 2752]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.2 ns   output offsets [64, 64, 128]
  some    l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns   output offsets [64, 96, 64]
  some    l1/prox_out/Float64/n4/normal                1.000–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 128]
  some    l1/prox_out/Float32/n2/normal                1.000–1.050   5.5 -> 5.5 ns   output offsets [64, 64, 64]

## xeon2640v4-julia1.13.4 (6 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 2016 of 2016

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3e | control |
|:--|:--|:--|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 1.00 [0.96–1.03] | 0.99 [0.96–1.02] | 1.00 |
| enet | block_loop | medium | 4 | 0.70 [0.67–0.74] | 0.70 [0.67–0.74] | 1.00 |
| enet | block_one | short | 2 | 1.04 [1.02–1.06] | 1.04 [1.02–1.06] | 1.00 |
| enet | prox_alloc | short | 14 | 0.98 [0.91–1.02] | 0.99 [0.92–1.02] | 1.00 |
| enet | prox_alloc | medium | 14 | 0.97 [0.92–1.01] | 0.98 [0.93–1.01] | 1.00 |
| enet | prox_alloc | large | 12 | 1.00 [0.98–1.01] | 1.00 [0.99–1.01] | 1.00 |
| enet | prox_g_out | short | 14 | 1.00 [1.00–1.01] | 1.00 [0.99–1.00] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | short | 14 | 0.86 [0.67–1.04] | 0.88 [0.70–1.06] | 1.00 |
| enet | prox_in | medium | 14 | 0.67 [0.67–0.69] | 0.67 [0.67–0.69] | 1.00 |
| enet | prox_in | large | 12 | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 1.00 |
| enet | prox_incopy | short | 14 | 0.87 [0.70–1.02] | 0.89 [0.70–1.06] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.69 [0.68–0.73] | 0.69 [0.68–0.73] | 1.00 |
| enet | prox_incopy | large | 12 | 0.72 [0.69–0.76] | 0.72 [0.69–0.76] | 1.00 |
| enet | prox_out | short | 14 | 0.97 [0.74–1.04] | 0.99 [0.75–1.06] | 1.00 |
| enet | prox_out | medium | 14 | 0.69 [0.67–0.74] | 0.75 [0.66–1.00] | 1.00 |
| enet | prox_out | large | 12 | 0.89 [0.67–0.99] | 1.00 [1.00–1.00] | 1.00 |
| enet | sparse_prox | large | 2 | 0.94 [0.93–0.94] | 0.94 [0.93–0.94] | 1.00 |
| enet | valprox_out | short | 14 | 0.98 [0.82–1.04] | 1.01 [0.83–1.07] | 1.00 |
| enet | valprox_out | medium | 14 | 0.76 [0.74–0.83] | 0.81 [0.74–1.00] | 1.00 |
| enet | valprox_out | large | 12 | 0.91 [0.75–0.99] | 1.00 [1.00–1.00] | 1.00 |
| enetw | block_loop | short | 4 | 1.02 [0.90–1.09] | 1.02 [0.90–1.08] | 1.00 |
| enetw | block_loop | medium | 4 | 0.84 [0.78–0.94] | 0.82 [0.78–0.87] | 1.00 |
| enetw | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.98 [0.93–1.01] | 0.98 [0.93–1.01] | 0.99 |
| enetw | prox_alloc | medium | 14 | 0.97 [0.90–1.00] | 0.97 [0.92–1.01] | 1.00 |
| enetw | prox_alloc | large | 12 | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_g_out | short | 14 | 1.00 [1.00–1.01] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | short | 14 | 0.87 [0.71–1.00] | 0.88 [0.71–1.04] | 1.00 |
| enetw | prox_in | medium | 14 | 0.68 [0.67–0.70] | 0.68 [0.67–0.70] | 1.00 |
| enetw | prox_in | large | 12 | 0.67 [0.67–0.68] | 0.67 [0.67–0.69] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.88 [0.73–1.01] | 0.88 [0.73–1.03] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.70 [0.68–0.72] | 0.70 [0.68–0.72] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 1.00 |
| enetw | prox_out | short | 14 | 0.95 [0.80–1.01] | 0.97 [0.79–1.04] | 1.00 |
| enetw | prox_out | medium | 14 | 0.74 [0.68–0.82] | 0.78 [0.68–1.00] | 1.00 |
| enetw | prox_out | large | 12 | 0.97 [0.80–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | sparse_prox | large | 2 | 0.94 [0.92–0.96] | 0.94 [0.92–0.96] | 1.00 |
| enetw | valprox_out | short | 14 | 0.96 [0.86–1.00] | 0.98 [0.88–1.02] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.81 [0.78–0.85] | 0.84 [0.78–1.00] | 1.00 |
| enetw | valprox_out | large | 12 | 0.98 [0.85–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 0.98 [0.94–1.04] | 0.99 [0.94–1.04] | 1.00 |
| l1 | block_loop | medium | 4 | 0.66 [0.65–0.68] | 0.66 [0.65–0.68] | 1.00 |
| l1 | block_one | short | 2 | 1.07 [1.07–1.07] | 1.04 [1.00–1.07] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.98 [0.92–1.03] | 1.00 [0.93–1.06] | 1.00 |
| l1 | prox_alloc | medium | 14 | 0.97 [0.92–0.99] | 0.98 [0.93–1.00] | 1.00 |
| l1 | prox_alloc | large | 12 | 1.00 [1.00–1.01] | 1.00 [0.98–1.01] | 1.00 |
| l1 | prox_g_out | short | 14 | 0.89 [0.80–1.04] | 0.90 [0.80–1.04] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.78 [0.78–0.79] | 0.78 [0.78–0.79] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.78 [0.78–0.80] | 0.78 [0.78–0.80] | 1.00 |
| l1 | prox_in | short | 14 | 0.84 [0.65–1.12] | 0.84 [0.65–1.12] | 1.00 |
| l1 | prox_in | medium | 14 | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 1.00 |
| l1 | prox_in | large | 12 | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.84 [0.69–1.08] | 0.86 [0.70–1.11] | 1.00 |
| l1 | prox_incopy | medium | 14 | 0.69 [0.67–0.73] | 0.69 [0.68–0.72] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 1.00 |
| l1 | prox_out | short | 14 | 0.95 [0.65–1.13] | 0.96 [0.65–1.17] | 1.00 |
| l1 | prox_out | medium | 14 | 0.67 [0.65–0.67] | 0.73 [0.65–1.00] | 1.00 |
| l1 | prox_out | large | 12 | 0.88 [0.67–0.99] | 1.00 [1.00–1.00] | 1.00 |
| l1 | sparse_prox | large | 2 | 0.95 [0.95–0.95] | 0.95 [0.95–0.95] | 1.00 |
| l1 | valprox_out | short | 14 | 0.97 [0.75–1.07] | 0.96 [0.76–1.07] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.74 [0.73–0.76] | 0.79 [0.73–1.00] | 1.00 |
| l1 | valprox_out | large | 12 | 0.91 [0.75–0.99] | 1.00 [1.00–1.00] | 1.00 |
| l1w | block_loop | short | 4 | 0.98 [0.91–1.03] | 0.98 [0.92–1.03] | 1.01 |
| l1w | block_loop | medium | 4 | 0.68 [0.67–0.69] | 0.68 [0.67–0.70] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 0.97 [0.92–1.01] | 0.98 [0.92–1.02] | 1.00 |
| l1w | prox_alloc | medium | 14 | 0.97 [0.92–1.00] | 0.97 [0.92–1.00] | 1.00 |
| l1w | prox_alloc | large | 12 | 1.00 [1.00–1.01] | 1.00 [0.99–1.00] | 1.00 |
| l1w | prox_g_out | short | 14 | 0.92 [0.85–1.05] | 0.91 [0.84–1.04] | 1.00 |
| l1w | prox_g_out | medium | 14 | 0.84 [0.83–0.84] | 0.84 [0.83–0.84] | 1.00 |
| l1w | prox_g_out | large | 12 | 0.84 [0.83–0.86] | 0.84 [0.83–0.86] | 1.00 |
| l1w | prox_in | short | 14 | 0.82 [0.68–1.04] | 0.82 [0.68–1.04] | 1.00 |
| l1w | prox_in | medium | 14 | 0.67 [0.67–0.68] | 0.67 [0.67–0.68] | 1.00 |
| l1w | prox_in | large | 12 | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.84 [0.70–1.02] | 0.86 [0.70–1.07] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.69 [0.68–0.71] | 0.69 [0.68–0.72] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.72 [0.69–0.80] | 0.72 [0.69–0.80] | 1.00 |
| l1w | prox_out | short | 14 | 0.93 [0.69–1.06] | 0.96 [0.71–1.10] | 0.99 |
| l1w | prox_out | medium | 14 | 0.71 [0.67–0.80] | 0.76 [0.67–1.00] | 1.00 |
| l1w | prox_out | large | 12 | 0.97 [0.79–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | sparse_prox | large | 2 | 0.95 [0.95–0.96] | 0.95 [0.94–0.96] | 1.00 |
| l1w | valprox_out | short | 14 | 0.96 [0.84–1.02] | 0.98 [0.84–1.06] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.83 [0.81–0.87] | 0.86 [0.82–1.00] | 1.00 |
| l1w | valprox_out | large | 12 | 0.98 [0.87–1.00] | 1.00 [1.00–1.00] | 1.00 |

A3M4: 4 of 1008 cells slower in every process, 471 faster in every process, 8 above 1.05 in some processes only, 62 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.125–1.125   7.1 -> 8.0 ns
  slower  l1/prox_in/Float64/n1/normal                 1.083–1.125   7.1 -> 8.0 ns
  slower  l1/block_one/Float32/n1/normal               1.071–1.071   4.1 -> 4.4 ns
  slower  enetw/block_loop/Float64/n16/normal          1.066–1.092   17.9 -> 19.3 ns
  some    l1/prox_g_out/Float32/n4/normal              0.927–1.080   13.3 -> 13.6 ns   output offsets [48, 48, 144, 48, 48, 48]
  some    l1/prox_incopy/Float64/n1/normal             1.037–1.079   11.2 -> 12.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/block_one/Float64/n1/normal               1.000–1.071   4.1 -> 4.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.034–1.069   8.6 -> 9.1 ns   output offsets [48, 48, 144, 48, 48, 48]
  some    enet/block_one/Float64/n1/normal             1.000–1.063   9.4 -> 10.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_in/Float64/n1/normal               0.983–1.053   17.0 -> 17.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float64/n1/normal              0.983–1.053   17.0 -> 17.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/valprox_out/Float64/n4/normal           0.974–1.052   28.4 -> 28.8 ns   output offsets [512, 384, 64, 1088, 64, 384]

A3e: 4 of 1008 cells slower in every process, 419 faster in every process, 16 above 1.05 in some processes only, 57 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.125–1.167   7.1 -> 8.0 ns
  slower  l1/prox_in/Float64/n1/normal                 1.125–1.125   7.1 -> 8.0 ns
  slower  l1/block_one/Float32/n1/normal               1.071–1.071   4.1 -> 4.4 ns
  slower  enetw/block_loop/Float64/n16/normal          1.066–1.084   17.9 -> 19.3 ns
  some    l1/prox_g_out/Float64/n4/normal              0.916–1.315   13.4 -> 12.4 ns   output offsets [512, 384, 64, 1088, 64, 384]
  some    l1/prox_out/Float32/n4/normal                1.034–1.103   8.6 -> 9.1 ns   output offsets [48, 48, 144, 48, 48, 48]
  some    enet/prox_in/Float64/n1/normal               1.034–1.088   17.0 -> 18.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float64/n1/normal              1.017–1.088   17.0 -> 18.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_g_out/Float64/n8/normal              0.856–1.085   22.8 -> 20.0 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    l1/prox_g_out/Float32/n4/normal              0.928–1.082   13.3 -> 13.0 ns   output offsets [48, 48, 144, 48, 48, 48]
  some    l1/prox_incopy/Float64/n1/normal             1.038–1.079   11.2 -> 12.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float32/n1/normal              1.038–1.078   15.3 -> 16.2 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_in/Float32/n1/normal               1.038–1.077   15.3 -> 16.2 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_incopy/Float64/n1/normal           1.028–1.072   21.6 -> 22.5 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/block_one/Float64/n1/normal               1.000–1.071   4.1 -> 4.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/valprox_out/Float64/n4/normal           1.004–1.067   28.4 -> 29.9 ns   output offsets [512, 384, 64, 1088, 64, 384]
  some    enet/block_one/Float64/n1/normal             1.000–1.063   9.4 -> 10.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float32/n4/normal              1.027–1.060   26.4 -> 27.5 ns   output offsets [48, 48, 144, 48, 48, 48]
  some    enet/valprox_out/Float64/n8/normal           0.995–1.052   30.5 -> 31.2 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    l1/prox_alloc/Float32/n8/normal              1.008–1.050   28.0 -> 28.6 ns   output offsets [128, 64, 64, 64, 64, 64]

## xeon4116-julia1.13.4 (6 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 2016 of 2016

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3e | control |
|:--|:--|:--|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 1.02 [0.96–1.09] | 1.02 [0.95–1.09] | 0.99 |
| enet | block_loop | medium | 4 | 0.86 [0.82–0.91] | 0.85 [0.83–0.90] | 1.00 |
| enet | block_one | short | 2 | 1.04 [1.03–1.06] | 1.02 [0.98–1.06] | 1.03 |
| enet | prox_alloc | short | 14 | 0.99 [0.97–1.01] | 0.99 [0.98–1.01] | 1.00 |
| enet | prox_alloc | medium | 14 | 1.01 [0.98–1.03] | 1.00 [0.98–1.03] | 1.00 |
| enet | prox_alloc | large | 12 | 1.03 [1.02–1.05] | 1.00 [1.00–1.01] | 1.00 |
| enet | prox_g_out | short | 14 | 1.00 [0.99–1.02] | 1.00 [0.98–1.03] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | short | 14 | 0.91 [0.82–1.03] | 0.92 [0.74–1.03] | 1.00 |
| enet | prox_in | medium | 14 | 0.77 [0.76–0.79] | 0.78 [0.77–0.80] | 1.00 |
| enet | prox_in | large | 12 | 0.77 [0.76–0.78] | 0.77 [0.76–0.79] | 1.00 |
| enet | prox_incopy | short | 14 | 0.93 [0.83–1.02] | 0.94 [0.83–1.06] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.79 [0.77–0.82] | 0.79 [0.77–0.83] | 1.00 |
| enet | prox_incopy | large | 12 | 0.82 [0.78–0.92] | 0.83 [0.78–0.91] | 1.01 |
| enet | prox_out | short | 14 | 0.98 [0.87–1.03] | 1.00 [0.90–1.06] | 1.00 |
| enet | prox_out | medium | 14 | 0.84 [0.82–0.90] | 0.87 [0.79–1.00] | 1.00 |
| enet | prox_out | large | 12 | 0.89 [0.81–1.02] | 1.00 [1.00–1.00] | 1.00 |
| enet | sparse_prox | large | 2 | 0.87 [0.87–0.87] | 0.87 [0.87–0.87] | 1.00 |
| enet | valprox_out | short | 14 | 0.99 [0.91–1.03] | 1.01 [0.92–1.05] | 1.00 |
| enet | valprox_out | medium | 14 | 0.88 [0.87–0.91] | 0.91 [0.87–1.00] | 1.00 |
| enet | valprox_out | large | 12 | 0.92 [0.86–1.02] | 1.00 [1.00–1.00] | 1.00 |
| enetw | block_loop | short | 4 | 0.98 [0.93–1.03] | 1.06 [1.01–1.17] | 1.00 |
| enetw | block_loop | medium | 4 | 0.91 [0.89–0.95] | 0.91 [0.88–0.94] | 1.00 |
| enetw | block_one | short | 2 | 1.05 [0.98–1.13] | 1.01 [0.99–1.02] | 0.98 |
| enetw | prox_alloc | short | 14 | 1.00 [0.99–1.00] | 0.99 [0.98–1.01] | 1.00 |
| enetw | prox_alloc | medium | 14 | 1.00 [0.99–1.01] | 1.00 [0.98–1.01] | 1.00 |
| enetw | prox_alloc | large | 12 | 1.01 [1.00–1.02] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_g_out | short | 14 | 1.01 [1.00–1.06] | 1.01 [0.99–1.02] | 1.02 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | short | 14 | 0.95 [0.87–1.04] | 0.94 [0.87–1.01] | 1.00 |
| enetw | prox_in | medium | 14 | 0.84 [0.84–0.86] | 0.84 [0.84–0.86] | 1.00 |
| enetw | prox_in | large | 12 | 0.84 [0.84–0.87] | 0.84 [0.84–0.87] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.95 [0.88–1.01] | 0.95 [0.88–1.00] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.86 [0.85–0.87] | 0.86 [0.84–0.89] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.87 [0.85–0.95] | 0.87 [0.85–0.95] | 1.00 |
| enetw | prox_out | short | 14 | 0.99 [0.95–1.04] | 0.97 [0.94–1.00] | 0.99 |
| enetw | prox_out | medium | 14 | 0.92 [0.90–0.95] | 0.94 [0.90–1.00] | 1.00 |
| enetw | prox_out | large | 12 | 0.96 [0.92–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | sparse_prox | large | 2 | 0.98 [0.98–0.99] | 0.98 [0.98–0.99] | 1.00 |
| enetw | valprox_out | short | 14 | 1.01 [0.97–1.05] | 0.99 [0.97–1.03] | 1.01 |
| enetw | valprox_out | medium | 14 | 0.95 [0.93–0.98] | 0.96 [0.93–1.00] | 1.00 |
| enetw | valprox_out | large | 12 | 0.97 [0.95–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 0.96 [0.90–1.01] | 0.99 [0.94–1.08] | 1.01 |
| l1 | block_loop | medium | 4 | 0.79 [0.77–0.81] | 0.78 [0.77–0.80] | 1.00 |
| l1 | block_one | short | 2 | 0.92 [0.92–0.92] | 0.92 [0.92–0.92] | 1.00 |
| l1 | prox_alloc | short | 14 | 1.00 [0.98–1.02] | 1.02 [0.99–1.06] | 1.00 |
| l1 | prox_alloc | medium | 14 | 1.01 [0.98–1.03] | 1.00 [0.99–1.02] | 1.00 |
| l1 | prox_alloc | large | 12 | 1.02 [1.01–1.03] | 1.00 [1.00–1.01] | 1.00 |
| l1 | prox_g_out | short | 14 | 0.95 [0.84–1.08] | 0.98 [0.92–1.03] | 0.99 |
| l1 | prox_g_out | medium | 14 | 0.92 [0.84–1.00] | 1.00 [0.99–1.00] | 1.01 |
| l1 | prox_g_out | large | 12 | 0.92 [0.84–1.00] | 1.00 [1.00–1.00] | 1.01 |
| l1 | prox_in | short | 14 | 0.83 [0.73–1.07] | 0.83 [0.73–1.05] | 1.00 |
| l1 | prox_in | medium | 14 | 0.73 [0.71–0.74] | 0.73 [0.71–0.74] | 1.00 |
| l1 | prox_in | large | 12 | 0.73 [0.71–0.75] | 0.73 [0.71–0.74] | 0.99 |
| l1 | prox_incopy | short | 14 | 0.85 [0.71–1.03] | 0.87 [0.72–1.03] | 0.98 |
| l1 | prox_incopy | medium | 14 | 0.76 [0.74–0.78] | 0.76 [0.74–0.78] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.79 [0.75–0.91] | 0.79 [0.74–0.91] | 1.00 |
| l1 | prox_out | short | 14 | 0.98 [0.83–1.15] | 1.00 [0.84–1.14] | 1.00 |
| l1 | prox_out | medium | 14 | 0.78 [0.75–0.81] | 0.83 [0.75–1.00] | 1.00 |
| l1 | prox_out | large | 12 | 0.86 [0.76–1.01] | 1.00 [1.00–1.00] | 1.00 |
| l1 | sparse_prox | large | 2 | 0.98 [0.96–0.99] | 0.98 [0.96–0.99] | 1.00 |
| l1 | valprox_out | short | 14 | 0.98 [0.89–1.04] | 0.99 [0.90–1.04] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.85 [0.82–0.90] | 0.89 [0.82–1.00] | 1.00 |
| l1 | valprox_out | large | 12 | 0.89 [0.83–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | block_loop | short | 4 | 0.98 [0.96–1.01] | 0.98 [0.96–1.02] | 1.00 |
| l1w | block_loop | medium | 4 | 0.86 [0.85–0.87] | 0.86 [0.85–0.87] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 1.00 [0.99–1.01] | 1.00 [0.99–1.02] | 1.00 |
| l1w | prox_alloc | medium | 14 | 1.00 [0.98–1.01] | 1.00 [0.98–1.01] | 1.00 |
| l1w | prox_alloc | large | 12 | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_g_out | short | 14 | 0.92 [0.83–1.03] | 0.92 [0.83–1.01] | 0.98 |
| l1w | prox_g_out | medium | 14 | 0.86 [0.83–0.89] | 0.86 [0.83–0.89] | 0.97 |
| l1w | prox_g_out | large | 12 | 0.87 [0.83–0.95] | 0.87 [0.83–0.95] | 0.98 |
| l1w | prox_in | short | 14 | 0.95 [0.87–1.02] | 0.93 [0.87–1.07] | 1.00 |
| l1w | prox_in | medium | 14 | 0.85 [0.85–0.87] | 0.84 [0.83–0.85] | 1.00 |
| l1w | prox_in | large | 12 | 0.86 [0.84–0.89] | 0.85 [0.83–0.90] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.95 [0.87–1.02] | 0.95 [0.85–1.07] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.86 [0.85–0.87] | 0.86 [0.84–0.89] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.88 [0.86–0.96] | 0.88 [0.84–0.96] | 1.00 |
| l1w | prox_out | short | 14 | 0.96 [0.88–1.02] | 1.00 [0.90–1.15] | 1.00 |
| l1w | prox_out | medium | 14 | 0.88 [0.85–0.92] | 0.91 [0.85–1.00] | 1.00 |
| l1w | prox_out | large | 12 | 0.96 [0.90–1.01] | 1.00 [1.00–1.00] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.00 [1.00–1.00] | 0.99 [0.99–0.99] | 0.99 |
| l1w | valprox_out | short | 14 | 0.98 [0.93–1.01] | 1.00 [0.93–1.03] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.93 [0.89–0.95] | 0.94 [0.89–1.00] | 1.00 |
| l1w | valprox_out | large | 12 | 0.97 [0.94–1.00] | 1.00 [1.00–1.00] | 1.00 |

A3M4: 0 of 1008 cells slower in every process, 225 faster in every process, 6 above 1.05 in some processes only, 212 undecided (control unclean)
  some    l1/block_loop/Float64/n16/normal             0.933–1.095   10.4 -> 9.7 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    l1/prox_in/Float32/n1/normal                 1.048–1.095   7.0 -> 7.5 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_in/Float64/n1/normal               0.984–1.087   15.6 -> 16.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/block_one/Float32/n1/normal               0.917–1.083   4.0 -> 3.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n100000/normal       1.040–1.059   35321.2 -> 37521.0 ns   output offsets [2752, 2752, 2816, 2752, 2816, 2752]
  some    l1w/valprox_out/Float64/n8/normal            0.982–1.050   21.2 -> 21.0 ns   output offsets [96, 96, 96, 96, 96, 96]

A3e: 0 of 1008 cells slower in every process, 169 faster in every process, 19 above 1.05 in some processes only, 210 undecided (control unclean)
  some    l1w/prox_alloc/Float32/n1/normal             0.988–1.694   28.7 -> 28.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float32/n4/normal             0.972–1.666   30.5 -> 30.6 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1w/prox_alloc/Float32/n8/normal             0.979–1.648   32.4 -> 32.3 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float32/n16/normal            0.960–1.539   36.1 -> 35.8 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    l1w/prox_alloc/Float32/n32/normal            0.983–1.286   41.8 -> 41.2 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    l1w/block_one/Float64/n1/normal              1.000–1.262   4.4 -> 4.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float32/n64/normal            0.976–1.237   54.2 -> 54.7 ns   output offsets [288, 288, 288, 576, 288, 576]
  some    enet/prox_alloc/Float64/n100000/normal       0.995–1.227   98010.6 -> 98231.3 ns   output offsets [1344, 1344, 1408, 1344, 1408, 1344]
  some    enet/prox_in/Float64/n1/normal               1.021–1.161   15.6 -> 16.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float32/n128/normal           0.964–1.128   80.2 -> 79.3 ns   output offsets [544, 544, 544, 544, 544, 544]
  some    l1/prox_in/Float64/n1/normal                 1.048–1.096   7.0 -> 7.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_in/Float32/n1/normal                 1.048–1.095   7.0 -> 7.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n1/normal              1.042–1.085   26.2 -> 27.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n4/normal              1.021–1.063   28.2 -> 29.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float32/n1/normal              1.017–1.060   26.9 -> 28.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float64/n2/normal             0.984–1.055   29.0 -> 29.8 ns   output offsets [48, 48, 48, 48, 48, 96]
  some    l1w/prox_alloc/Float64/n16/normal            0.987–1.055   40.5 -> 40.8 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    l1w/valprox_out/Float32/n1/normal            1.009–1.054   14.9 -> 15.2 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n2048/normal           1.006–1.051   1823.8 -> 1859.3 ns   output offsets [1024, 3776, 2048, 3968, 2880, 3776]
