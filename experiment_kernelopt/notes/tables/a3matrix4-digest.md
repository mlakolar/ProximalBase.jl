
## epyc7513a-julia1.13.4 (3 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 3024 of 3024

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3dr | A3n | control |
|:--|:--|:--|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 0.99 [0.94–1.00] | 0.99 [0.97–1.00] | 1.00 [1.00–1.00] | 0.99 |
| enet | block_loop | medium | 4 | 0.74 [0.68–0.80] | 0.73 [0.68–0.80] | 1.00 [1.00–1.01] | 1.00 |
| enet | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_alloc | short | 14 | 0.98 [0.87–1.12] | 0.99 [0.86–1.01] | 1.00 [0.98–1.01] | 0.99 |
| enet | prox_alloc | medium | 14 | 0.90 [0.75–0.97] | 0.94 [0.88–1.06] | 1.01 [0.95–1.18] | 0.96 |
| enet | prox_alloc | large | 12 | 0.93 [0.76–1.03] | 0.99 [0.86–1.04] | 0.97 [0.81–1.06] | 0.98 |
| enet | prox_g_out | short | 14 | 1.00 [0.96–1.01] | 1.00 [0.98–1.03] | 1.01 [0.95–1.08] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [0.99–1.00] | 1.00 [1.00–1.00] | 1.00 [0.99–1.01] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 1.00 |
| enet | prox_in | short | 14 | 0.86 [0.72–1.00] | 0.87 [0.72–1.03] | 1.00 [0.98–1.03] | 1.00 |
| enet | prox_in | medium | 14 | 0.70 [0.69–0.71] | 0.70 [0.69–0.71] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | large | 12 | 0.69 [0.69–0.69] | 0.69 [0.69–0.69] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_incopy | short | 14 | 0.88 [0.74–1.01] | 0.89 [0.73–1.02] | 1.00 [0.98–1.02] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_incopy | large | 12 | 0.73 [0.72–0.75] | 0.73 [0.72–0.75] | 1.00 [1.00–1.01] | 1.00 |
| enet | prox_out | short | 14 | 0.95 [0.74–1.00] | 0.99 [0.73–1.06] | 1.00 [0.98–1.03] | 1.00 |
| enet | prox_out | medium | 14 | 0.71 [0.68–0.80] | 0.77 [0.68–1.00] | 1.00 [1.00–1.01] | 1.00 |
| enet | prox_out | large | 12 | 0.97 [0.71–1.52] | 0.99 [0.88–1.01] | 1.00 [0.98–1.03] | 1.00 |
| enet | sparse_prox | large | 2 | 1.09 [1.09–1.10] | 1.09 [1.09–1.10] | 1.00 [0.94–1.05] | 1.02 |
| enet | valprox_out | short | 14 | 0.98 [0.88–1.03] | 1.00 [0.88–1.05] | 1.01 [0.98–1.02] | 1.01 |
| enet | valprox_out | medium | 14 | 0.79 [0.76–0.89] | 0.84 [0.76–1.01] | 1.00 [0.98–1.01] | 1.00 |
| enet | valprox_out | large | 12 | 0.99 [0.78–1.36] | 1.00 [0.98–1.02] | 1.00 [0.98–1.08] | 0.99 |
| enetw | block_loop | short | 4 | 1.01 [0.95–1.14] | 1.01 [0.90–1.11] | 1.01 [1.00–1.02] | 1.00 |
| enetw | block_loop | medium | 4 | 0.88 [0.85–0.93] | 1.01 [0.99–1.03] | 1.00 [0.99–1.00] | 0.99 |
| enetw | block_one | short | 2 | 0.98 [0.97–1.00] | 0.98 [0.97–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.98 [0.93–1.01] | 0.99 [0.95–1.03] | 1.00 [0.99–1.01] | 0.99 |
| enetw | prox_alloc | medium | 14 | 0.95 [0.89–1.04] | 0.96 [0.91–1.02] | 1.00 [0.95–1.04] | 0.96 |
| enetw | prox_alloc | large | 12 | 0.98 [0.86–1.18] | 1.03 [0.84–1.19] | 0.97 [0.81–1.15] | 0.98 |
| enetw | prox_g_out | short | 14 | 0.99 [0.98–1.01] | 1.00 [0.98–1.02] | 1.00 [0.99–1.02] | 1.02 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [0.99–1.01] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_in | short | 14 | 0.92 [0.82–1.00] | 0.92 [0.82–1.00] | 1.00 [0.98–1.02] | 1.00 |
| enetw | prox_in | medium | 14 | 0.78 [0.78–0.80] | 0.78 [0.78–0.80] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | large | 12 | 0.79 [0.78–0.83] | 0.79 [0.78–0.83] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.92 [0.83–1.00] | 0.93 [0.83–1.00] | 0.99 [0.98–1.00] | 0.99 |
| enetw | prox_incopy | medium | 14 | 0.80 [0.79–0.82] | 0.80 [0.79–0.82] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.81 [0.80–0.85] | 0.81 [0.79–0.84] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_out | short | 14 | 0.98 [0.87–1.02] | 1.00 [0.88–1.05] | 0.99 [0.98–1.00] | 0.99 |
| enetw | prox_out | medium | 14 | 0.83 [0.72–0.98] | 0.84 [0.73–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_out | large | 12 | 1.04 [0.93–1.24] | 1.00 [0.98–1.01] | 1.00 [0.97–1.01] | 1.00 |
| enetw | sparse_prox | large | 2 | 1.01 [0.97–1.06] | 0.94 [0.94–0.95] | 0.99 [0.99–1.00] | 1.00 |
| enetw | valprox_out | short | 14 | 0.96 [0.83–1.02] | 0.99 [0.85–1.05] | 1.00 [0.98–1.02] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.89 [0.83–0.98] | 0.90 [0.83–1.01] | 1.00 [0.96–1.00] | 1.00 |
| enetw | valprox_out | large | 12 | 1.03 [0.95–1.16] | 1.00 [0.99–1.00] | 1.00 [0.98–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 0.97 [0.92–1.00] | 0.97 [0.92–1.00] | 0.97 [0.92–1.00] | 1.00 |
| l1 | block_loop | medium | 4 | 0.69 [0.67–0.77] | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 1.00 |
| l1 | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.99 [0.95–1.01] | 1.02 [0.96–1.04] | 0.98 [0.95–1.00] | 1.00 |
| l1 | prox_alloc | medium | 14 | 0.96 [0.91–1.04] | 0.97 [0.94–1.02] | 0.94 [0.84–1.02] | 0.96 |
| l1 | prox_alloc | large | 12 | 1.01 [0.87–1.12] | 0.98 [0.86–1.03] | 1.01 [0.84–1.11] | 0.93 |
| l1 | prox_g_out | short | 14 | 1.01 [0.93–1.05] | 1.00 [0.93–1.05] | 1.00 [0.93–1.05] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.99 [0.98–1.00] | 0.98 [0.98–0.99] | 0.99 [0.98–1.00] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.98 [0.97–0.98] | 0.98 [0.97–0.98] | 0.98 [0.97–0.99] | 1.00 |
| l1 | prox_in | short | 14 | 0.82 [0.68–1.00] | 0.82 [0.68–1.00] | 0.84 [0.68–1.06] | 0.99 |
| l1 | prox_in | medium | 14 | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 1.00 |
| l1 | prox_in | large | 12 | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.87 [0.70–1.03] | 0.89 [0.72–1.09] | 0.87 [0.70–1.03] | 1.00 |
| l1 | prox_incopy | medium | 14 | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 0.73 [0.71–0.74] | 1.00 |
| l1 | prox_out | short | 14 | 0.95 [0.77–1.05] | 1.01 [0.79–1.14] | 0.97 [0.79–1.06] | 1.00 |
| l1 | prox_out | medium | 14 | 0.68 [0.67–0.79] | 0.74 [0.67–1.00] | 0.68 [0.67–0.79] | 1.00 |
| l1 | prox_out | large | 12 | 0.91 [0.67–1.19] | 1.03 [0.99–1.17] | 0.91 [0.67–1.23] | 1.04 |
| l1 | sparse_prox | large | 2 | 1.02 [1.01–1.03] | 1.00 [1.00–1.01] | 1.01 [1.01–1.01] | 1.00 |
| l1 | valprox_out | short | 14 | 0.96 [0.86–1.02] | 1.00 [0.87–1.08] | 0.96 [0.87–1.03] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.77 [0.74–0.87] | 0.82 [0.74–1.00] | 0.77 [0.74–0.86] | 1.00 |
| l1 | valprox_out | large | 12 | 0.93 [0.75–1.13] | 1.02 [1.00–1.14] | 0.93 [0.75–1.14] | 1.02 |
| l1w | block_loop | short | 4 | 0.99 [0.90–1.09] | 0.96 [0.90–1.00] | 0.99 [0.93–1.05] | 1.00 |
| l1w | block_loop | medium | 4 | 0.72 [0.68–0.78] | 0.71 [0.68–0.76] | 0.72 [0.68–0.78] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 0.98 [0.91–1.00] | 1.00 [0.94–1.02] | 0.98 [0.93–1.00] | 0.99 |
| l1w | prox_alloc | medium | 14 | 1.00 [0.87–1.14] | 0.99 [0.89–1.12] | 1.00 [0.88–1.10] | 0.97 |
| l1w | prox_alloc | large | 12 | 1.06 [0.94–1.15] | 0.99 [0.86–1.05] | 1.06 [0.89–1.14] | 0.96 |
| l1w | prox_g_out | short | 14 | 1.06 [1.00–1.10] | 0.99 [0.91–1.04] | 1.02 [0.91–1.10] | 1.00 |
| l1w | prox_g_out | medium | 14 | 1.10 [1.10–1.11] | 0.98 [0.96–1.00] | 1.03 [0.96–1.11] | 1.00 |
| l1w | prox_g_out | large | 12 | 1.10 [1.08–1.11] | 0.98 [0.96–1.00] | 1.02 [0.96–1.10] | 1.00 |
| l1w | prox_in | short | 14 | 0.90 [0.76–1.00] | 0.89 [0.76–1.00] | 0.89 [0.76–1.00] | 1.00 |
| l1w | prox_in | medium | 14 | 0.72 [0.71–0.73] | 0.71 [0.71–0.73] | 0.72 [0.71–0.73] | 1.00 |
| l1w | prox_in | large | 12 | 0.71 [0.71–0.72] | 0.71 [0.71–0.71] | 0.71 [0.71–0.72] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.90 [0.77–1.00] | 0.92 [0.78–1.04] | 0.90 [0.78–1.00] | 0.99 |
| l1w | prox_incopy | medium | 14 | 0.74 [0.72–0.76] | 0.74 [0.72–0.76] | 0.74 [0.72–0.76] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.75 [0.73–0.77] | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 1.00 |
| l1w | prox_out | short | 14 | 0.97 [0.84–1.04] | 1.00 [0.85–1.08] | 0.96 [0.84–1.00] | 1.00 |
| l1w | prox_out | medium | 14 | 0.79 [0.67–0.99] | 0.80 [0.68–1.00] | 0.79 [0.67–0.99] | 1.00 |
| l1w | prox_out | large | 12 | 1.02 [0.99–1.16] | 1.00 [1.00–1.00] | 1.01 [0.99–1.10] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.03 [1.02–1.04] | 1.04 [1.04–1.05] | 1.04 [1.04–1.04] | 1.00 |
| l1w | valprox_out | short | 14 | 0.97 [0.88–1.00] | 1.01 [0.91–1.08] | 0.97 [0.88–1.02] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.89 [0.82–1.00] | 0.90 [0.83–1.00] | 0.89 [0.82–1.00] | 1.00 |
| l1w | valprox_out | large | 12 | 1.01 [0.98–1.08] | 1.00 [0.99–1.01] | 1.00 [0.99–1.05] | 1.00 |

A3M4: 3 of 1008 cells slower in every process, 394 faster in every process, 38 above 1.05 in some processes only, 75 undecided (control unclean)
  slower  enet/valprox_out/Float32/n1048576/normal     1.309–1.404   167212.9 -> 229417.1 ns
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 11.2 ns
  slower  enetw/prox_out/Float32/n1048576/normal       1.070–1.128   142216.1 -> 153392.5 ns
  some    enetw/prox_out/Float32/n16384/normal         0.929–1.296   1834.2 -> 1704.2 ns   output offsets [3968, 128, 256]
  some    enetw/prox_out/Float32/n32768/normal         0.973–1.274   3729.7 -> 4729.5 ns   output offsets [2176, 64, 64]
  some    l1/prox_out/Float32/n32768/normal            0.671–1.242   3432.6 -> 4280.3 ns   output offsets [2176, 64, 64]
  some    l1/prox_out/Float32/n16384/normal            0.673–1.214   1689.5 -> 1137.2 ns   output offsets [3968, 128, 256]
  some    l1w/prox_alloc/Float32/n64/normal            0.950–1.211   28.6 -> 32.7 ns   output offsets [288, 288, 288]
  some    l1/valprox_out/Float32/n32768/normal         0.755–1.175   4603.1 -> 5029.2 ns   output offsets [2176, 64, 64]
  some    enetw/prox_incopy/Float64/n4/normal          0.985–1.163   17.8 -> 17.6 ns   output offsets [64, 192, 384]
  some    l1/valprox_out/Float32/n16384/normal         0.754–1.163   2287.1 -> 1724.0 ns   output offsets [3968, 128, 256]
  some    enetw/valprox_out/Float32/n32768/normal      0.982–1.154   5987.6 -> 6722.0 ns   output offsets [2176, 64, 64]
  some    enetw/prox_out/Float64/n65536/normal         1.045–1.150   18146.5 -> 19813.6 ns   output offsets [128, 64, 64]
  some    enetw/prox_incopy/Float32/n1/normal          0.983–1.121   15.8 -> 15.6 ns   output offsets [96, 64, 64]
  some    enetw/prox_out/Float32/n1/normal             1.024–1.119   11.5 -> 11.7 ns   output offsets [96, 64, 64]
  some    enetw/valprox_out/Float64/n65536/normal      1.042–1.111   29367.8 -> 31398.0 ns   output offsets [128, 64, 64]
  some    l1w/prox_g_out/Float64/n256/normal           0.970–1.109   325.0 -> 359.2 ns   output offsets [2048, 1472, 2048]
  some    l1w/prox_g_out/Float64/n1000/normal          0.962–1.107   1243.0 -> 1375.5 ns   output offsets [1600, 4032, 1984]
  some    l1w/prox_g_out/Float64/n65536/normal         0.951–1.107   84207.6 -> 91087.8 ns   output offsets [128, 64, 64]
  some    l1w/prox_g_out/Float64/n1024/normal          0.962–1.106   1274.5 -> 1408.5 ns   output offsets [1600, 3200, 2432]
  some    l1w/prox_g_out/Float64/n128/normal           0.972–1.106   166.9 -> 184.4 ns   output offsets [1088, 1088, 1088]
  some    l1w/prox_g_out/Float64/n32768/normal         0.952–1.106   41506.4 -> 45677.5 ns   output offsets [128, 128, 64]
  some    l1w/prox_g_out/Float64/n100000/normal        0.960–1.106   125998.3 -> 139037.4 ns   output offsets [1408, 1344, 1344]
  some    enetw/valprox_out/Float32/n262144/normal     1.036–1.104   59556.3 -> 63134.4 ns   output offsets [128, 64, 64]
  some    l1w/prox_g_out/Float64/n16384/normal         0.960–1.103   20790.7 -> 22852.2 ns   output offsets [3968, 2944, 0]
  some    l1w/prox_g_out/Float64/n8192/normal          0.958–1.103   10322.8 -> 11299.4 ns   output offsets [2240, 2240, 896]
  some    l1w/prox_g_out/Float64/n4096/normal          0.962–1.102   5089.5 -> 5610.6 ns   output offsets [1408, 1280, 2432]
  some    enetw/valprox_out/Float32/n1048576/normal    1.049–1.102   233187.6 -> 244722.9 ns   output offsets [128, 64, 64]

A3dr: 8 of 1008 cells slower in every process, 352 faster in every process, 33 above 1.05 in some processes only, 31 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 10.9 ns
  slower  l1/prox_out/Float64/n4/normal                1.100–1.100   5.5 -> 6.0 ns
  slower  l1/prox_incopy/Float32/n1/normal             1.061–1.107   8.7 -> 9.6 ns
  slower  l1/prox_incopy/Float32/n2/normal             1.059–1.086   9.1 -> 9.8 ns
  slower  l1w/valprox_out/Float64/n1/normal            1.056–1.083   9.8 -> 10.7 ns
  slower  l1/prox_out/Float32/n1/normal                1.055–1.056   4.9 -> 5.2 ns
  slower  l1w/prox_out/Float64/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_out/Float32/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  some    enetw/valprox_out/Float64/n1/normal          1.001–1.139   17.9 -> 18.7 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float64/n4/normal           1.021–1.121   18.2 -> 18.8 ns   output offsets [64, 192, 384]
  some    l1w/prox_g_out/Float64/n1000/normal          0.962–1.108   1243.0 -> 1198.3 ns   output offsets [1600, 4032, 1984]
  some    l1w/prox_g_out/Float64/n128/normal           0.965–1.106   166.9 -> 162.0 ns   output offsets [1088, 1088, 1088]
  some    l1w/prox_g_out/Float64/n1024/normal          0.961–1.105   1274.5 -> 1233.7 ns   output offsets [1600, 3200, 2432]
  some    l1w/prox_g_out/Float64/n4096/normal          0.959–1.105   5089.5 -> 4962.7 ns   output offsets [1408, 1280, 2432]
  some    l1w/prox_g_out/Float64/n2048/normal          0.958–1.103   2556.8 -> 2508.1 ns   output offsets [0, 4032, 2816]
  some    l1w/prox_g_out/Float64/n100000/normal        0.961–1.101   125998.3 -> 121196.2 ns   output offsets [1408, 1344, 1344]
  some    l1/prox_out/Float32/n4/normal                1.050–1.100   5.5 -> 5.7 ns   output offsets [48, 48, 48]
  some    l1w/prox_g_out/Float64/n32/normal            0.976–1.100   43.7 -> 42.7 ns   output offsets [288, 288, 288]
  some    l1w/prox_g_out/Float64/n8192/normal          0.958–1.098   10322.8 -> 9904.3 ns   output offsets [2240, 2240, 896]
  some    l1w/prox_g_out/Float64/n262144/normal        0.961–1.098   331179.8 -> 318412.3 ns   output offsets [0, 0, 0]
  some    l1w/prox_g_out/Float64/n16384/normal         0.960–1.096   20790.7 -> 19984.6 ns   output offsets [3968, 2944, 0]
  some    enetw/prox_alloc/Float64/n32/normal          0.967–1.095   32.9 -> 31.8 ns   output offsets [288, 288, 288]
  some    l1w/prox_g_out/Float64/n256/normal           0.962–1.092   325.0 -> 313.2 ns   output offsets [2048, 1472, 2048]
  some    l1w/block_one/Float64/n1/normal              1.000–1.091   3.0 -> 3.0 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float64/n16/normal            0.977–1.091   24.0 -> 23.5 ns   output offsets [160, 160, 160]
  some    enetw/valprox_out/Float64/n4/normal          0.993–1.088   20.9 -> 21.1 ns   output offsets [64, 192, 384]
  some    l1w/prox_g_out/Float64/n65536/normal         0.954–1.078   84207.6 -> 80374.0 ns   output offsets [128, 64, 64]
  some    l1w/prox_g_out/Float64/n32768/normal         0.962–1.077   41506.4 -> 40056.0 ns   output offsets [128, 128, 64]
  some    enetw/valprox_out/Float64/n16/normal         0.981–1.076   25.0 -> 24.6 ns   output offsets [160, 160, 160]
  some    enetw/block_one/Float64/n1/normal            0.970–1.066   6.0 -> 5.8 ns   output offsets [64, 64, 64]
  some    l1/prox_incopy/Float32/n4/normal             1.002–1.062   10.8 -> 11.2 ns   output offsets [48, 48, 48]
  some    l1w/valprox_out/Float64/n8/normal            1.016–1.061   13.5 -> 14.2 ns   output offsets [96, 96, 96]
  some    l1w/prox_g_out/Float64/n4/normal             1.000–1.059   9.3 -> 9.3 ns   output offsets [64, 192, 384]

A3n: 0 of 1008 cells slower in every process, 199 faster in every process, 32 above 1.05 in some processes only, 38 undecided (control unclean)
  some    l1/prox_out/Float32/n32768/normal            0.671–1.248   3432.6 -> 4285.0 ns   output offsets [2176, 64, 64]
  some    l1/prox_out/Float32/n16384/normal            0.673–1.237   1689.5 -> 1137.3 ns   output offsets [3968, 128, 256]
  some    enet/valprox_out/Float32/n1048576/normal     0.972–1.216   167212.9 -> 176998.0 ns   output offsets [128, 64, 64]
  some    l1/valprox_out/Float32/n32768/normal         0.755–1.176   4603.1 -> 5017.7 ns   output offsets [2176, 64, 64]
  some    l1/valprox_out/Float32/n16384/normal         0.753–1.172   2287.1 -> 1721.7 ns   output offsets [3968, 128, 256]
  some    enet/prox_incopy/Float32/n8/normal           1.000–1.156   17.5 -> 17.5 ns   output offsets [64, 64, 64]
  some    enet/prox_g_out/Float32/n64/normal           0.980–1.139   43.3 -> 43.3 ns   output offsets [288, 288, 288]
  some    enetw/prox_alloc/Float64/n32/normal          0.984–1.129   32.9 -> 32.5 ns   output offsets [288, 288, 288]
  some    enetw/prox_alloc/Float32/n1048576/normal     0.985–1.114   160702.5 -> 170741.0 ns   output offsets [128, 64, 64]
  some    l1w/prox_g_out/Float64/n256/normal           0.962–1.109   325.0 -> 359.5 ns   output offsets [2048, 1472, 2048]
  some    l1w/prox_g_out/Float64/n1000/normal          0.962–1.108   1243.0 -> 1375.5 ns   output offsets [1600, 4032, 1984]
  some    l1w/prox_g_out/Float64/n32768/normal         0.963–1.107   41506.4 -> 45682.8 ns   output offsets [128, 128, 64]
  some    l1w/prox_g_out/Float64/n128/normal           0.965–1.106   166.9 -> 184.4 ns   output offsets [1088, 1088, 1088]
  some    l1w/prox_g_out/Float64/n1024/normal          0.961–1.106   1274.5 -> 1408.8 ns   output offsets [1600, 3200, 2432]
  some    l1w/prox_g_out/Float64/n4096/normal          0.962–1.105   5089.5 -> 5606.4 ns   output offsets [1408, 1280, 2432]
  some    l1w/prox_g_out/Float64/n16384/normal         0.960–1.104   20790.7 -> 22793.9 ns   output offsets [3968, 2944, 0]
  some    l1w/prox_g_out/Float64/n2048/normal          0.959–1.103   2556.8 -> 2807.8 ns   output offsets [0, 4032, 2816]
  some    l1w/prox_g_out/Float64/n100000/normal        0.963–1.103   125998.3 -> 138725.4 ns   output offsets [1408, 1344, 1344]
  some    l1w/prox_g_out/Float64/n262144/normal        0.960–1.100   331179.8 -> 364345.6 ns   output offsets [0, 0, 0]
  some    l1w/prox_g_out/Float64/n32/normal            0.975–1.100   43.7 -> 48.1 ns   output offsets [288, 288, 288]
  some    l1w/prox_g_out/Float64/n8192/normal          0.957–1.098   10322.8 -> 11333.9 ns   output offsets [2240, 2240, 896]
  some    l1w/prox_g_out/Float64/n1048576/normal       0.964–1.092   1348507.0 -> 1464446.0 ns   output offsets [0, 0, 0]
  some    l1w/block_loop/Float32/n4/normal             1.000–1.091   6.0 -> 6.3 ns   output offsets [48, 48, 48]
  some    l1w/block_one/Float64/n1/normal              1.000–1.091   3.0 -> 3.0 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float64/n16/normal            0.977–1.091   24.0 -> 26.2 ns   output offsets [160, 160, 160]

## epyc7513b-julia1.13.4 (3 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 3024 of 3024

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3dr | A3n | control |
|:--|:--|:--|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 1.01 [1.00–1.03] | 0.99 [0.95–1.03] | 1.00 [1.00–1.00] | 1.00 |
| enet | block_loop | medium | 4 | 0.74 [0.68–0.81] | 0.73 [0.69–0.81] | 1.00 [1.00–1.00] | 1.00 |
| enet | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_alloc | short | 14 | 0.98 [0.87–1.00] | 0.99 [0.91–1.03] | 1.00 [0.99–1.03] | 0.99 |
| enet | prox_alloc | medium | 14 | 0.92 [0.89–0.96] | 0.95 [0.88–1.04] | 1.01 [0.98–1.05] | 0.95 |
| enet | prox_alloc | large | 12 | 0.95 [0.88–1.11] | 1.00 [0.90–1.06] | 1.00 [0.88–1.08] | 1.00 |
| enet | prox_g_out | short | 14 | 1.00 [0.98–1.03] | 1.00 [0.97–1.04] | 1.00 [0.98–1.01] | 1.01 |
| enet | prox_g_out | medium | 14 | 1.00 [0.99–1.00] | 1.00 [1.00–1.00] | 1.00 [0.99–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | short | 14 | 0.87 [0.71–1.03] | 0.87 [0.71–1.03] | 1.00 [0.98–1.01] | 1.00 |
| enet | prox_in | medium | 14 | 0.70 [0.69–0.71] | 0.70 [0.69–0.71] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | large | 12 | 0.69 [0.69–0.69] | 0.69 [0.69–0.69] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_incopy | short | 14 | 0.88 [0.73–1.00] | 0.88 [0.73–1.00] | 0.99 [0.98–1.00] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_incopy | large | 12 | 0.73 [0.72–0.75] | 0.73 [0.72–0.75] | 1.00 [1.00–1.01] | 1.00 |
| enet | prox_out | short | 14 | 0.95 [0.73–1.00] | 0.99 [0.77–1.06] | 0.99 [0.98–1.02] | 1.00 |
| enet | prox_out | medium | 14 | 0.72 [0.68–0.82] | 0.78 [0.68–1.00] | 1.00 [0.99–1.01] | 1.00 |
| enet | prox_out | large | 12 | 0.99 [0.70–1.56] | 0.98 [0.90–1.03] | 1.00 [0.88–1.17] | 1.02 |
| enet | sparse_prox | large | 2 | 1.09 [1.09–1.09] | 1.09 [1.09–1.09] | 1.00 [0.99–1.00] | 1.00 |
| enet | valprox_out | short | 14 | 0.97 [0.88–1.01] | 1.00 [0.89–1.04] | 1.00 [0.99–1.01] | 1.00 |
| enet | valprox_out | medium | 14 | 0.79 [0.76–0.88] | 0.84 [0.76–1.00] | 1.00 [0.99–1.01] | 1.00 |
| enet | valprox_out | large | 12 | 1.02 [0.78–1.42] | 0.99 [0.94–1.03] | 1.01 [0.93–1.18] | 1.02 |
| enetw | block_loop | short | 4 | 1.00 [0.91–1.11] | 1.01 [0.89–1.14] | 1.00 [0.98–1.03] | 1.00 |
| enetw | block_loop | medium | 4 | 0.88 [0.86–0.91] | 0.96 [0.88–1.01] | 1.00 [1.00–1.01] | 1.00 |
| enetw | block_one | short | 2 | 1.00 [1.00–1.00] | 0.98 [0.97–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.99 [0.95–1.08] | 1.00 [0.96–1.02] | 1.00 [0.99–1.02] | 0.99 |
| enetw | prox_alloc | medium | 14 | 0.96 [0.89–1.04] | 0.97 [0.91–1.09] | 1.00 [0.95–1.04] | 0.95 |
| enetw | prox_alloc | large | 12 | 0.98 [0.85–1.04] | 0.97 [0.81–1.05] | 0.95 [0.81–1.05] | 0.95 |
| enetw | prox_g_out | short | 14 | 1.00 [0.98–1.02] | 1.01 [0.98–1.03] | 1.00 [0.98–1.02] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [0.99–1.01] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [0.99–1.01] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | short | 14 | 0.92 [0.82–1.02] | 0.92 [0.81–1.02] | 1.00 [0.98–1.01] | 1.00 |
| enetw | prox_in | medium | 14 | 0.78 [0.78–0.81] | 0.78 [0.78–0.80] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_in | large | 12 | 0.78 [0.78–0.83] | 0.78 [0.78–0.83] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.93 [0.83–1.02] | 0.93 [0.82–1.02] | 1.01 [0.99–1.04] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.80 [0.79–0.81] | 0.80 [0.79–0.82] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.81 [0.80–0.85] | 0.81 [0.80–0.85] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_out | short | 14 | 0.97 [0.86–1.02] | 0.99 [0.88–1.05] | 1.00 [0.98–1.00] | 1.00 |
| enetw | prox_out | medium | 14 | 0.83 [0.73–0.96] | 0.85 [0.73–1.01] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_out | large | 12 | 1.04 [0.97–1.18] | 1.00 [0.98–1.01] | 1.00 [0.97–1.01] | 1.00 |
| enetw | sparse_prox | large | 2 | 0.95 [0.95–0.96] | 1.00 [0.95–1.06] | 1.00 [0.99–1.00] | 1.00 |
| enetw | valprox_out | short | 14 | 0.96 [0.82–1.00] | 0.98 [0.84–1.03] | 1.00 [0.99–1.01] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.89 [0.83–0.99] | 0.90 [0.83–1.01] | 1.00 [1.00–1.00] | 1.00 |
| enetw | valprox_out | large | 12 | 1.03 [0.98–1.10] | 1.00 [0.99–1.00] | 1.00 [0.98–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 1.00 [0.96–1.04] | 0.99 [0.96–1.05] | 0.98 [0.96–1.00] | 1.00 |
| l1 | block_loop | medium | 4 | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 1.00 |
| l1 | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.98 [0.95–1.00] | 1.01 [0.90–1.04] | 0.98 [0.95–1.00] | 0.99 |
| l1 | prox_alloc | medium | 14 | 0.95 [0.85–1.03] | 0.96 [0.90–1.04] | 0.95 [0.89–1.03] | 0.97 |
| l1 | prox_alloc | large | 12 | 1.03 [0.85–1.24] | 0.99 [0.83–1.26] | 1.04 [0.83–1.26] | 0.96 |
| l1 | prox_g_out | short | 14 | 1.00 [0.93–1.05] | 1.00 [0.93–1.05] | 1.00 [0.93–1.05] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.98 [0.98–1.00] | 0.98 [0.97–0.98] | 0.98 [0.97–0.98] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.98 [0.97–0.99] | 0.98 [0.97–0.98] | 0.98 [0.97–0.98] | 1.00 |
| l1 | prox_in | short | 14 | 0.83 [0.68–1.00] | 0.83 [0.68–1.01] | 0.83 [0.68–1.00] | 0.99 |
| l1 | prox_in | medium | 14 | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 1.00 |
| l1 | prox_in | large | 12 | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.87 [0.70–1.00] | 0.89 [0.72–1.09] | 0.87 [0.70–1.00] | 1.00 |
| l1 | prox_incopy | medium | 14 | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 |
| l1 | prox_out | short | 14 | 0.96 [0.79–1.05] | 1.01 [0.79–1.10] | 0.95 [0.77–1.05] | 1.00 |
| l1 | prox_out | medium | 14 | 0.68 [0.67–0.77] | 0.74 [0.67–1.00] | 0.68 [0.67–0.77] | 1.00 |
| l1 | prox_out | large | 12 | 0.89 [0.67–1.21] | 1.00 [0.96–1.01] | 0.90 [0.67–1.21] | 1.01 |
| l1 | sparse_prox | large | 2 | 1.00 [1.00–1.01] | 1.01 [1.01–1.01] | 1.03 [1.01–1.04] | 1.01 |
| l1 | valprox_out | short | 14 | 0.96 [0.86–1.03] | 1.00 [0.88–1.05] | 0.96 [0.87–1.03] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.77 [0.74–0.84] | 0.82 [0.74–1.00] | 0.77 [0.74–0.85] | 1.00 |
| l1 | valprox_out | large | 12 | 0.92 [0.75–1.15] | 1.00 [0.96–1.02] | 0.93 [0.76–1.15] | 1.01 |
| l1w | block_loop | short | 4 | 0.98 [0.93–1.05] | 0.97 [0.93–1.00] | 0.99 [0.93–1.05] | 1.00 |
| l1w | block_loop | medium | 4 | 0.72 [0.67–0.77] | 0.72 [0.68–0.76] | 0.71 [0.68–0.75] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.04 [1.00–1.09] | 1.00 |
| l1w | prox_alloc | short | 14 | 0.98 [0.81–1.10] | 0.99 [0.94–1.01] | 0.98 [0.93–1.00] | 0.99 |
| l1w | prox_alloc | medium | 14 | 0.98 [0.90–1.12] | 0.97 [0.87–1.07] | 0.98 [0.88–1.13] | 0.98 |
| l1w | prox_alloc | large | 12 | 1.07 [0.87–1.32] | 0.97 [0.79–1.04] | 1.04 [0.93–1.18] | 0.96 |
| l1w | prox_g_out | short | 14 | 0.98 [0.91–1.03] | 1.02 [0.95–1.10] | 1.02 [0.92–1.10] | 1.03 |
| l1w | prox_g_out | medium | 14 | 0.96 [0.96–0.97] | 1.05 [1.00–1.11] | 1.03 [0.96–1.11] | 1.05 |
| l1w | prox_g_out | large | 12 | 0.96 [0.95–0.96] | 1.05 [0.99–1.11] | 1.03 [0.95–1.11] | 1.04 |
| l1w | prox_in | short | 14 | 0.89 [0.76–1.00] | 0.90 [0.75–1.00] | 0.89 [0.76–1.00] | 1.00 |
| l1w | prox_in | medium | 14 | 0.72 [0.71–0.73] | 0.72 [0.71–0.73] | 0.72 [0.71–0.73] | 1.00 |
| l1w | prox_in | large | 12 | 0.71 [0.71–0.71] | 0.71 [0.71–0.71] | 0.71 [0.71–0.71] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.90 [0.77–1.00] | 0.94 [0.78–1.12] | 0.91 [0.77–1.03] | 0.99 |
| l1w | prox_incopy | medium | 14 | 0.74 [0.72–0.76] | 0.74 [0.72–0.76] | 0.74 [0.72–0.76] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 1.00 |
| l1w | prox_out | short | 14 | 0.97 [0.81–1.04] | 1.00 [0.84–1.05] | 0.97 [0.83–1.04] | 1.00 |
| l1w | prox_out | medium | 14 | 0.80 [0.68–0.99] | 0.80 [0.67–1.00] | 0.80 [0.67–0.99] | 1.00 |
| l1w | prox_out | large | 12 | 1.00 [0.95–1.02] | 1.00 [0.99–1.01] | 1.00 [0.95–1.02] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.02 [1.00–1.04] | 1.01 [1.00–1.01] | 1.00 [0.99–1.01] | 1.01 |
| l1w | valprox_out | short | 14 | 0.97 [0.90–1.02] | 1.00 [0.91–1.07] | 0.96 [0.89–1.01] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.89 [0.82–1.00] | 0.90 [0.82–1.00] | 0.89 [0.82–1.00] | 1.00 |
| l1w | valprox_out | large | 12 | 1.00 [0.98–1.02] | 1.00 [0.97–1.00] | 1.00 [0.97–1.02] | 1.00 |

A3M4: 9 of 1008 cells slower in every process, 379 faster in every process, 25 above 1.05 in some processes only, 67 undecided (control unclean)
  slower  l1w/prox_alloc/Float32/n1048576/normal       1.151–1.388   154405.5 -> 191109.5 ns
  slower  enetw/prox_out/Float32/n65536/normal         1.144–1.187   8190.4 -> 9919.8 ns
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.111   9.8 -> 10.9 ns
  slower  l1/prox_out/Float32/n65536/normal            1.107–1.171   7117.9 -> 8656.5 ns
  slower  l1/valprox_out/Float32/n65536/normal         1.075–1.137   9633.4 -> 10945.1 ns
  slower  enetw/prox_out/Float64/n32768/normal         1.070–1.271   9340.8 -> 10718.1 ns
  slower  enetw/valprox_out/Float32/n65536/normal      1.069–1.110   13661.0 -> 14977.4 ns
  slower  enetw/prox_out/Float32/n262144/normal        1.065–1.213   34938.7 -> 38877.2 ns
  slower  enetw/valprox_out/Float64/n32768/normal      1.062–1.175   14374.7 -> 15828.4 ns
  some    enetw/prox_out/Float64/n16384/normal         0.978–1.337   3457.1 -> 3392.3 ns   output offsets [1728, 128, 2368]
  some    l1/valprox_out/Float64/n16384/normal         0.752–1.281   4520.0 -> 3403.3 ns   output offsets [1728, 128, 2368]
  some    enet/prox_alloc/Float32/n1048576/normal      0.968–1.258   128521.5 -> 149847.0 ns   output offsets [64, 64, 128]
  some    l1/prox_out/Float64/n16384/normal            0.670–1.247   3373.4 -> 2259.0 ns   output offsets [1728, 128, 2368]
  some    enetw/valprox_out/Float64/n16384/normal      0.988–1.245   5714.1 -> 5652.6 ns   output offsets [1728, 128, 2368]
  some    enetw/prox_alloc/Float32/n1048576/normal     0.998–1.240   150503.5 -> 166087.5 ns   output offsets [64, 64, 128]
  some    l1/prox_out/Float32/n32768/normal            0.670–1.211   3367.2 -> 2263.2 ns   output offsets [64, 512, 1664]
  some    l1w/prox_alloc/Float32/n64/normal            0.937–1.200   28.1 -> 31.4 ns   output offsets [288, 288, 288]
  some    enetw/prox_out/Float32/n32768/normal         0.962–1.171   3534.3 -> 3400.8 ns   output offsets [64, 512, 1664]
  some    l1/valprox_out/Float32/n32768/normal         0.754–1.168   4517.2 -> 3411.1 ns   output offsets [64, 512, 1664]
  some    enet/prox_out/Float32/n32768/normal          0.700–1.119   3408.8 -> 2391.2 ns   output offsets [64, 512, 1664]
  some    enetw/prox_alloc/Float64/n32/normal          0.969–1.109   32.9 -> 35.1 ns   output offsets [288, 288, 288]
  some    enet/sparse_prox/Float32/n65536/tenth        0.948–1.093   26197.4 -> 28356.3 ns   output offsets [128, 64, 64]
  some    l1w/block_one/Float64/n1/normal              1.000–1.091   3.0 -> 3.0 ns   output offsets [64, 64, 64]
  some    l1w/block_loop/Float32/n4/normal             1.045–1.091   6.0 -> 6.3 ns   output offsets [48, 48, 48]
  some    enetw/sparse_prox/Float64/n65536/tenth       0.909–1.087   21510.0 -> 20907.8 ns   output offsets [128, 64, 128]
  some    enetw/prox_alloc/Float32/n2/normal           0.991–1.085   25.1 -> 27.3 ns   output offsets [64, 128, 64]
  some    l1w/prox_out/Float32/n32768/normal           0.993–1.081   3399.9 -> 3376.6 ns   output offsets [64, 512, 1664]
  some    enetw/valprox_out/Float32/n32768/normal      0.971–1.077   5825.2 -> 5657.3 ns   output offsets [64, 512, 1664]
  some    enetw/prox_incopy/Float32/n4/normal          0.986–1.076   18.6 -> 18.6 ns   output offsets [48, 48, 48]
  some    l1w/valprox_out/Float32/n32768/normal        0.989–1.073   6284.2 -> 6213.7 ns   output offsets [64, 512, 1664]
  some    enetw/prox_alloc/Float32/n1/normal           1.000–1.068   24.4 -> 24.4 ns   output offsets [64, 128, 64]
  some    enet/valprox_out/Float32/n32768/normal       0.776–1.067   4646.8 -> 3624.6 ns   output offsets [64, 512, 1664]
  some    l1w/valprox_out/Float64/n16384/normal        0.998–1.056   6222.3 -> 6226.0 ns   output offsets [1728, 128, 2368]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.2 ns   output offsets [64, 192, 128]

A3dr: 7 of 1008 cells slower in every process, 345 faster in every process, 15 above 1.05 in some processes only, 45 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 11.2 ns
  slower  enet/sparse_prox/Float32/n65536/tenth        1.087–1.106   26197.4 -> 28591.9 ns
  slower  l1w/valprox_out/Float64/n1/normal            1.056–1.083   9.8 -> 10.4 ns
  slower  l1/prox_out/Float32/n1/normal                1.056–1.111   4.9 -> 5.2 ns
  slower  l1w/prox_out/Float64/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_out/Float32/n1/normal               1.052–1.053   5.2 -> 5.5 ns
  slower  l1/prox_out/Float32/n2/normal                1.050–1.100   5.5 -> 5.7 ns
  some    enetw/prox_out/Float64/n8/normal             1.000–1.480   13.7 -> 13.7 ns   output offsets [96, 96, 96]
  some    enetw/prox_out/Float64/n16/normal            1.000–1.409   14.8 -> 14.8 ns   output offsets [160, 160, 160]
  some    enetw/prox_out/Float64/n32/normal            0.969–1.249   17.5 -> 16.9 ns   output offsets [288, 288, 288]
  some    l1w/prox_incopy/Float32/n2/normal            0.823–1.114   9.8 -> 10.7 ns   output offsets [64, 128, 64]
  some    l1/prox_out/Float64/n4/normal                1.050–1.100   5.5 -> 6.0 ns   output offsets [64, 192, 128]
  some    enetw/valprox_out/Float32/n8/normal          1.006–1.082   21.3 -> 22.0 ns   output offsets [64, 64, 832]
  some    l1/valprox_out/Float32/n65536/normal         1.004–1.079   9633.4 -> 10088.4 ns   output offsets [128, 64, 64]
  some    l1w/prox_out/Float32/n262144/normal          0.997–1.070   37124.7 -> 37094.6 ns   output offsets [64, 64, 128]
  some    enetw/prox_out/Float32/n262144/normal        0.998–1.064   34938.7 -> 36140.0 ns   output offsets [64, 64, 128]
  some    enetw/prox_out/Float64/n64/normal            0.874–1.063   24.6 -> 21.7 ns   output offsets [544, 544, 544]
  some    enet/valprox_out/Float64/n4/normal           1.030–1.060   18.3 -> 19.0 ns   output offsets [64, 192, 128]
  some    l1/prox_alloc/Float32/n2/normal              1.025–1.060   18.1 -> 18.6 ns   output offsets [64, 128, 64]
  some    l1/prox_in/Float32/n1/normal                 1.000–1.056   4.9 -> 4.9 ns   output offsets [64, 128, 64]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.5 ns   output offsets [64, 192, 128]
  some    l1/prox_incopy/Float32/n4/normal             1.025–1.051   10.9 -> 11.2 ns   output offsets [48, 48, 48]

A3n: 3 of 1008 cells slower in every process, 201 faster in every process, 21 above 1.05 in some processes only, 33 undecided (control unclean)
  slower  l1/prox_out/Float32/n65536/normal            1.172–1.225   7117.9 -> 8675.2 ns
  slower  l1w/prox_alloc/Float32/n1048576/normal       1.116–1.347   154405.5 -> 193323.5 ns
  slower  l1/valprox_out/Float32/n65536/normal         1.073–1.186   9633.4 -> 10946.6 ns
  some    enetw/prox_incopy/Float64/n4/normal          0.985–1.546   17.8 -> 17.5 ns   output offsets [64, 192, 128]
  some    l1/valprox_out/Float64/n16384/normal         0.754–1.284   4520.0 -> 3418.3 ns   output offsets [1728, 128, 2368]
  some    enet/prox_alloc/Float32/n1048576/normal      1.003–1.268   128521.5 -> 134788.5 ns   output offsets [64, 64, 128]
  some    enetw/prox_alloc/Float32/n1048576/normal     0.998–1.255   150503.5 -> 157306.0 ns   output offsets [64, 64, 128]
  some    l1/prox_out/Float64/n16384/normal            0.670–1.253   3373.4 -> 2259.1 ns   output offsets [1728, 128, 2368]
  some    enetw/prox_alloc/Float32/n2/normal           0.999–1.219   25.1 -> 25.2 ns   output offsets [64, 128, 64]
  some    l1/prox_out/Float32/n32768/normal            0.670–1.211   3367.2 -> 2266.8 ns   output offsets [64, 512, 1664]
  some    l1w/valprox_out/Float64/n32/normal           0.963–1.146   21.1 -> 20.3 ns   output offsets [288, 288, 288]
  some    l1w/prox_out/Float32/n32768/normal           0.993–1.107   3399.9 -> 3376.6 ns   output offsets [64, 512, 1664]
  some    l1/valprox_out/Float32/n32768/normal         0.753–1.104   4517.2 -> 3409.4 ns   output offsets [64, 512, 1664]
  some    l1w/block_loop/Float32/n4/normal             1.000–1.091   6.0 -> 6.3 ns   output offsets [48, 48, 48]
  some    l1w/block_one/Float64/n1/normal              1.000–1.091   3.0 -> 3.3 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float32/n32/normal          1.008–1.089   21.0 -> 21.2 ns   output offsets [160, 160, 160]
  some    enet/valprox_out/Float32/n128/normal         0.993–1.077   32.4 -> 32.1 ns   output offsets [544, 544, 544]
  some    enetw/prox_out/Float32/n16384/normal         0.998–1.076   1757.0 -> 1768.3 ns   output offsets [1920, 384, 2624]
  some    l1w/valprox_out/Float32/n32768/normal        0.990–1.074   6284.2 -> 6220.1 ns   output offsets [64, 512, 1664]
  some    l1w/valprox_out/Float64/n16384/normal        0.998–1.056   6222.3 -> 6210.6 ns   output offsets [1728, 128, 2368]
  some    enetw/prox_out/Float32/n128/normal           1.004–1.056   24.1 -> 24.3 ns   output offsets [544, 544, 544]
  some    enetw/prox_alloc/Float64/n8/normal           0.992–1.053   26.9 -> 26.9 ns   output offsets [96, 96, 96]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.2 ns   output offsets [64, 192, 128]
  some    l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns   output offsets [64, 128, 64]

## xeon2640v4-julia1.13.4 (6 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 3024 of 3024

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3dr | A3n | control |
|:--|:--|:--|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 1.01 [0.96–1.05] | 1.00 [0.96–1.04] | 1.00 [1.00–1.02] | 0.99 |
| enet | block_loop | medium | 4 | 0.69 [0.66–0.74] | 0.69 [0.66–0.74] | 1.00 [1.00–1.00] | 1.00 |
| enet | block_one | short | 2 | 1.03 [1.03–1.03] | 1.04 [1.03–1.05] | 1.03 [1.00–1.06] | 1.03 |
| enet | prox_alloc | short | 14 | 0.97 [0.91–1.01] | 0.99 [0.91–1.02] | 1.00 [0.98–1.01] | 1.00 |
| enet | prox_alloc | medium | 14 | 0.97 [0.92–1.00] | 0.97 [0.93–0.99] | 1.00 [1.00–1.02] | 1.00 |
| enet | prox_alloc | large | 12 | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 1.00 [0.99–1.00] | 1.00 |
| enet | prox_g_out | short | 14 | 1.00 [0.99–1.01] | 1.00 [1.00–1.03] | 1.00 [1.00–1.03] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | short | 14 | 0.85 [0.66–1.02] | 0.86 [0.71–1.02] | 0.99 [0.95–1.00] | 1.00 |
| enet | prox_in | medium | 14 | 0.67 [0.67–0.69] | 0.67 [0.67–0.69] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | large | 12 | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_incopy | short | 14 | 0.87 [0.69–1.01] | 0.88 [0.72–1.03] | 1.00 [0.97–1.01] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.69 [0.68–0.73] | 0.69 [0.68–0.73] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 0.72 [0.69–0.76] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_out | short | 14 | 0.96 [0.75–1.02] | 1.00 [0.78–1.08] | 1.00 [0.98–1.00] | 1.00 |
| enet | prox_out | medium | 14 | 0.69 [0.66–0.75] | 0.69 [0.67–0.76] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_out | large | 12 | 0.89 [0.68–0.99] | 0.89 [0.68–0.99] | 1.00 [1.00–1.00] | 1.00 |
| enet | sparse_prox | large | 2 | 0.93 [0.93–0.94] | 0.93 [0.93–0.94] | 1.00 [1.00–1.00] | 1.00 |
| enet | valprox_out | short | 14 | 0.97 [0.81–1.01] | 1.01 [0.84–1.06] | 0.99 [0.99–1.00] | 1.00 |
| enet | valprox_out | medium | 14 | 0.76 [0.74–0.83] | 0.76 [0.74–0.84] | 1.00 [0.99–1.00] | 1.00 |
| enet | valprox_out | large | 12 | 0.92 [0.75–0.99] | 0.92 [0.75–0.99] | 1.00 [1.00–1.00] | 1.00 |
| enetw | block_loop | short | 4 | 1.03 [0.90–1.08] | 1.03 [0.90–1.08] | 0.99 [0.98–1.00] | 1.00 |
| enetw | block_loop | medium | 4 | 0.82 [0.78–0.90] | 0.81 [0.78–0.89] | 1.00 [0.99–1.00] | 1.00 |
| enetw | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.97 [0.92–1.01] | 0.98 [0.92–1.02] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_alloc | medium | 14 | 0.97 [0.90–0.99] | 0.98 [0.91–1.00] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_alloc | large | 12 | 1.00 [0.99–1.01] | 1.00 [0.99–1.02] | 1.00 [0.99–1.00] | 1.00 |
| enetw | prox_g_out | short | 14 | 1.00 [0.99–1.02] | 1.00 [0.99–1.01] | 1.00 [0.99–1.02] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | short | 14 | 0.87 [0.71–1.00] | 0.87 [0.71–1.02] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_in | medium | 14 | 0.68 [0.67–0.70] | 0.68 [0.67–0.70] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | large | 12 | 0.67 [0.67–0.68] | 0.67 [0.67–0.68] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.87 [0.73–1.00] | 0.87 [0.73–1.00] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.69 [0.68–0.72] | 0.70 [0.68–0.72] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_out | short | 14 | 0.95 [0.81–1.00] | 0.98 [0.82–1.05] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_out | medium | 14 | 0.74 [0.68–0.82] | 0.75 [0.68–0.82] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_out | large | 12 | 0.97 [0.80–1.00] | 0.97 [0.80–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | sparse_prox | large | 2 | 0.94 [0.92–0.96] | 0.94 [0.93–0.96] | 1.00 [1.00–1.00] | 1.00 |
| enetw | valprox_out | short | 14 | 0.97 [0.86–1.01] | 0.99 [0.88–1.03] | 1.00 [0.99–1.01] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.81 [0.78–0.87] | 0.82 [0.78–0.87] | 1.00 [1.00–1.00] | 1.00 |
| enetw | valprox_out | large | 12 | 0.98 [0.85–1.00] | 0.98 [0.85–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 0.99 [0.94–1.06] | 0.99 [0.94–1.04] | 0.99 [0.94–1.04] | 1.01 |
| l1 | block_loop | medium | 4 | 0.66 [0.64–0.68] | 0.66 [0.64–0.68] | 0.66 [0.64–0.68] | 1.00 |
| l1 | block_one | short | 2 | 1.07 [1.07–1.07] | 1.07 [1.07–1.07] | 1.04 [1.00–1.07] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.98 [0.92–1.03] | 1.01 [0.93–1.07] | 0.98 [0.92–1.03] | 1.00 |
| l1 | prox_alloc | medium | 14 | 0.97 [0.93–1.00] | 0.98 [0.93–1.00] | 0.97 [0.92–1.00] | 1.00 |
| l1 | prox_alloc | large | 12 | 1.00 [0.99–1.01] | 1.00 [0.99–1.02] | 1.00 [0.98–1.00] | 1.00 |
| l1 | prox_g_out | short | 14 | 0.90 [0.80–1.04] | 0.90 [0.80–1.04] | 0.90 [0.80–1.04] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.78 [0.78–0.79] | 0.78 [0.78–0.79] | 0.78 [0.78–0.80] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.78 [0.78–0.79] | 0.78 [0.78–0.79] | 0.78 [0.78–0.79] | 1.00 |
| l1 | prox_in | short | 14 | 0.85 [0.67–1.13] | 0.85 [0.66–1.12] | 0.84 [0.67–1.13] | 1.00 |
| l1 | prox_in | medium | 14 | 0.67 [0.65–0.67] | 0.67 [0.66–0.67] | 0.67 [0.66–0.67] | 1.00 |
| l1 | prox_in | large | 12 | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.85 [0.69–1.06] | 0.86 [0.70–1.11] | 0.84 [0.69–1.06] | 1.00 |
| l1 | prox_incopy | medium | 14 | 0.69 [0.67–0.73] | 0.69 [0.68–0.73] | 0.69 [0.67–0.73] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 1.00 |
| l1 | prox_out | short | 14 | 0.95 [0.66–1.12] | 1.01 [0.66–1.25] | 0.95 [0.65–1.13] | 1.00 |
| l1 | prox_out | medium | 14 | 0.67 [0.65–0.68] | 0.67 [0.66–0.67] | 0.67 [0.65–0.68] | 1.00 |
| l1 | prox_out | large | 12 | 0.88 [0.67–0.99] | 0.88 [0.67–0.99] | 0.88 [0.67–0.99] | 1.00 |
| l1 | sparse_prox | large | 2 | 0.95 [0.95–0.95] | 0.95 [0.95–0.96] | 0.95 [0.95–0.96] | 1.00 |
| l1 | valprox_out | short | 14 | 0.96 [0.77–1.03] | 0.98 [0.77–1.08] | 0.95 [0.76–1.03] | 0.99 |
| l1 | valprox_out | medium | 14 | 0.74 [0.73–0.76] | 0.74 [0.72–0.77] | 0.74 [0.73–0.76] | 1.00 |
| l1 | valprox_out | large | 12 | 0.91 [0.75–0.99] | 0.91 [0.75–0.99] | 0.91 [0.75–0.99] | 1.00 |
| l1w | block_loop | short | 4 | 0.97 [0.90–1.03] | 0.97 [0.90–1.03] | 0.97 [0.90–1.03] | 1.00 |
| l1w | block_loop | medium | 4 | 0.68 [0.67–0.71] | 0.68 [0.67–0.70] | 0.68 [0.67–0.70] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 0.97 [0.92–1.01] | 0.98 [0.92–1.02] | 0.98 [0.92–1.01] | 1.00 |
| l1w | prox_alloc | medium | 14 | 0.97 [0.92–1.01] | 0.98 [0.92–1.00] | 0.97 [0.93–1.01] | 1.00 |
| l1w | prox_alloc | large | 12 | 1.00 [0.98–1.00] | 1.00 [1.00–1.01] | 1.00 [0.99–1.01] | 1.00 |
| l1w | prox_g_out | short | 14 | 0.92 [0.84–1.06] | 0.92 [0.84–1.04] | 0.92 [0.84–1.04] | 1.01 |
| l1w | prox_g_out | medium | 14 | 0.83 [0.83–0.84] | 0.83 [0.83–0.84] | 0.83 [0.83–0.84] | 1.00 |
| l1w | prox_g_out | large | 12 | 0.84 [0.83–0.86] | 0.84 [0.83–0.86] | 0.84 [0.83–0.86] | 1.00 |
| l1w | prox_in | short | 14 | 0.82 [0.68–1.04] | 0.82 [0.68–1.04] | 0.82 [0.68–1.02] | 1.00 |
| l1w | prox_in | medium | 14 | 0.67 [0.67–0.68] | 0.67 [0.67–0.68] | 0.67 [0.67–0.68] | 1.00 |
| l1w | prox_in | large | 12 | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.84 [0.70–1.02] | 0.86 [0.70–1.08] | 0.84 [0.70–1.02] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.72 [0.69–0.80] | 0.72 [0.69–0.80] | 0.72 [0.69–0.80] | 1.00 |
| l1w | prox_out | short | 14 | 0.92 [0.70–1.04] | 1.01 [0.72–1.16] | 0.93 [0.70–1.04] | 1.00 |
| l1w | prox_out | medium | 14 | 0.72 [0.67–0.80] | 0.72 [0.67–0.80] | 0.72 [0.67–0.80] | 1.00 |
| l1w | prox_out | large | 12 | 0.97 [0.79–1.00] | 0.97 [0.79–1.00] | 0.97 [0.79–1.00] | 1.00 |
| l1w | sparse_prox | large | 2 | 0.95 [0.95–0.96] | 0.95 [0.95–0.96] | 0.95 [0.95–0.96] | 1.00 |
| l1w | valprox_out | short | 14 | 0.96 [0.84–1.02] | 1.00 [0.85–1.09] | 0.96 [0.84–1.02] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.83 [0.81–0.87] | 0.84 [0.82–0.87] | 0.83 [0.81–0.87] | 1.00 |
| l1w | valprox_out | large | 12 | 0.98 [0.87–1.00] | 0.98 [0.87–1.00] | 0.98 [0.87–1.00] | 1.00 |

A3M4: 3 of 1008 cells slower in every process, 472 faster in every process, 9 above 1.05 in some processes only, 64 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.125–1.125   7.1 -> 8.0 ns
  slower  l1/block_one/Float32/n1/normal               1.071–1.071   4.1 -> 4.4 ns
  slower  enetw/block_loop/Float32/n4/normal           1.059–1.100   15.0 -> 16.1 ns
  some    l1/prox_incopy/Float64/n1/normal             1.038–1.079   11.4 -> 12.1 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    l1/block_one/Float64/n1/normal               1.000–1.072   4.1 -> 4.4 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    enet/valprox_out/Float32/n8/normal           0.989–1.069   27.8 -> 27.8 ns   output offsets [64, 64, 64, 256, 640, 256]
  some    l1/prox_out/Float32/n4/normal                1.034–1.069   8.6 -> 9.1 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_alloc/Float32/n1/normal              1.012–1.068   25.0 -> 25.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float64/n16384/normal        0.967–1.068   14002.9 -> 14590.9 ns   output offsets [1920, 3776, 2752, 2112, 0, 3392]
  some    enet/block_one/Float64/n1/normal             1.000–1.063   9.4 -> 9.7 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    enet/valprox_out/Float32/n1/normal           0.993–1.062   22.6 -> 22.9 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float32/n4/normal             0.972–1.055   28.4 -> 28.6 ns   output offsets [48, 48, 48, 48, 48, 48]

A3dr: 7 of 1008 cells slower in every process, 465 faster in every process, 23 above 1.05 in some processes only, 76 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.208–1.250   7.1 -> 8.6 ns
  slower  l1/prox_incopy/Float32/n1/normal             1.113–1.136   13.0 -> 14.5 ns
  slower  l1/prox_out/Float32/n4/normal                1.103–1.172   8.6 -> 9.7 ns
  slower  enet/prox_out/Float32/n1/normal              1.077–1.098   15.3 -> 16.5 ns
  slower  l1/block_one/Float32/n1/normal               1.071–1.071   4.1 -> 4.4 ns
  slower  enetw/block_loop/Float32/n4/normal           1.059–1.092   15.0 -> 16.0 ns
  slower  l1w/valprox_out/Float32/n1/normal            1.059–1.080   14.7 -> 15.9 ns
  some    l1/prox_alloc/Float32/n1/normal              1.044–1.125   25.0 -> 26.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float64/n1/normal              1.034–1.103   17.1 -> 18.1 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float32/n2/normal              1.040–1.093   25.7 -> 27.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/valprox_out/Float32/n1/normal           1.020–1.086   22.6 -> 24.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_incopy/Float32/n2/normal             1.049–1.085   14.1 -> 14.9 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n4/normal            1.003–1.081   35.2 -> 35.8 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_out/Float64/n2/normal              1.016–1.081   18.4 -> 19.0 ns   output offsets [384, 288, 48, 48, 48, 48]
  some    enet/prox_out/Float64/n4/normal              1.031–1.079   18.7 -> 19.5 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_incopy/Float64/n1/normal             1.038–1.079   11.4 -> 12.1 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    l1w/valprox_out/Float32/n4/normal            1.045–1.076   19.5 -> 20.9 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/block_one/Float64/n1/normal               1.000–1.071   4.1 -> 4.4 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float32/n2/normal              1.018–1.071   16.8 -> 17.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float32/n4/normal             0.998–1.070   28.4 -> 28.9 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_in/Float64/n1/normal               0.983–1.069   17.1 -> 17.3 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float32/n8/normal              1.033–1.067   18.0 -> 18.9 ns   output offsets [64, 64, 64, 256, 640, 256]
  some    enet/valprox_out/Float32/n8/normal           1.017–1.065   27.8 -> 28.5 ns   output offsets [64, 64, 64, 256, 640, 256]
  some    enetw/prox_g_out/Float32/n64/normal          0.998–1.064   95.1 -> 95.2 ns   output offsets [288, 288, 576, 288, 288, 288]
  some    enet/block_one/Float64/n1/normal             1.000–1.063   9.4 -> 9.9 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    enet/prox_incopy/Float64/n1/normal           0.986–1.057   21.3 -> 22.0 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float32/n1/normal             1.006–1.053   26.2 -> 26.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_alloc/Float64/n16/normal          0.974–1.052   48.7 -> 48.2 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    l1w/valprox_out/Float32/n2/normal            1.017–1.052   17.1 -> 17.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_incopy/Float32/n4/normal             0.965–1.050   17.2 -> 17.9 ns   output offsets [48, 48, 48, 48, 48, 48]

A3n: 2 of 1008 cells slower in every process, 271 faster in every process, 8 above 1.05 in some processes only, 32 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.125–1.125   7.1 -> 8.0 ns
  slower  l1/block_one/Float32/n1/normal               1.071–1.072   4.1 -> 4.4 ns
  some    l1w/prox_alloc/Float32/n4/normal             0.991–1.096   28.4 -> 29.1 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_out/Float32/n64/normal             0.988–1.083   24.0 -> 24.1 ns   output offsets [288, 288, 576, 288, 288, 288]
  some    enetw/prox_out/Float64/n32/normal            0.968–1.082   29.8 -> 29.5 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    l1/prox_incopy/Float64/n1/normal             1.038–1.079   11.4 -> 12.1 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    l1/block_one/Float64/n1/normal               0.999–1.071   4.1 -> 4.1 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.034–1.069   8.6 -> 9.1 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_alloc/Float32/n1/normal              1.008–1.058   25.0 -> 25.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/valprox_out/Float32/n256/normal         0.994–1.053   89.2 -> 89.1 ns   output offsets [1088, 2176, 1088, 1088, 2240, 1088]

## xeon4116-julia1.13.4 (6 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 3024 of 3024

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3dr | A3n | control |
|:--|:--|:--|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 0.99 [0.95–1.07] | 1.00 [0.96–1.07] | 1.00 [0.99–1.02] | 1.00 |
| enet | block_loop | medium | 4 | 0.85 [0.83–0.88] | 0.85 [0.84–0.88] | 1.00 [1.00–1.00] | 1.00 |
| enet | block_one | short | 2 | 0.98 [0.93–1.03] | 0.97 [0.93–1.02] | 1.01 [1.00–1.02] | 1.01 |
| enet | prox_alloc | short | 14 | 0.99 [0.96–1.01] | 1.00 [0.96–1.02] | 1.01 [0.99–1.02] | 1.00 |
| enet | prox_alloc | medium | 14 | 1.01 [0.98–1.03] | 1.01 [0.98–1.03] | 1.00 [0.98–1.01] | 1.00 |
| enet | prox_alloc | large | 12 | 1.03 [1.01–1.05] | 1.03 [1.02–1.06] | 1.00 [0.99–1.01] | 1.00 |
| enet | prox_g_out | short | 14 | 1.01 [0.96–1.03] | 1.00 [0.99–1.01] | 0.99 [0.96–1.01] | 0.99 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | short | 14 | 0.91 [0.82–1.04] | 0.91 [0.74–1.07] | 1.00 [1.00–1.02] | 1.00 |
| enet | prox_in | medium | 14 | 0.77 [0.76–0.79] | 0.77 [0.75–0.79] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | large | 12 | 0.78 [0.76–0.79] | 0.77 [0.76–0.80] | 1.01 [1.00–1.03] | 1.00 |
| enet | prox_incopy | short | 14 | 0.93 [0.83–1.00] | 0.94 [0.83–1.05] | 1.00 [0.99–1.02] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.79 [0.78–0.82] | 0.79 [0.77–0.83] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_incopy | large | 12 | 0.82 [0.78–0.93] | 0.82 [0.78–0.93] | 1.00 [1.00–1.01] | 1.00 |
| enet | prox_out | short | 14 | 0.96 [0.87–1.03] | 1.04 [0.92–1.14] | 1.00 [0.96–1.02] | 1.01 |
| enet | prox_out | medium | 14 | 0.84 [0.81–0.87] | 0.84 [0.79–0.89] | 1.00 [0.99–1.00] | 1.00 |
| enet | prox_out | large | 12 | 0.89 [0.81–1.02] | 0.89 [0.81–1.02] | 1.00 [1.00–1.00] | 1.00 |
| enet | sparse_prox | large | 2 | 0.89 [0.88–0.90] | 0.88 [0.87–0.90] | 1.00 [1.00–1.01] | 1.01 |
| enet | valprox_out | short | 14 | 0.98 [0.92–1.01] | 1.02 [0.92–1.09] | 0.99 [0.97–1.03] | 0.99 |
| enet | valprox_out | medium | 14 | 0.88 [0.87–0.91] | 0.89 [0.87–0.92] | 1.00 [1.00–1.00] | 1.00 |
| enet | valprox_out | large | 12 | 0.92 [0.86–1.01] | 0.92 [0.86–1.01] | 1.00 [1.00–1.00] | 1.00 |
| enetw | block_loop | short | 4 | 1.04 [1.00–1.11] | 0.99 [0.92–1.04] | 1.01 [0.99–1.05] | 0.99 |
| enetw | block_loop | medium | 4 | 0.91 [0.90–0.94] | 0.92 [0.90–0.95] | 1.00 [1.00–1.00] | 1.00 |
| enetw | block_one | short | 2 | 1.01 [0.97–1.04] | 0.99 [0.97–1.02] | 1.01 [0.97–1.05] | 0.98 |
| enetw | prox_alloc | short | 14 | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_alloc | medium | 14 | 1.00 [0.98–1.01] | 1.00 [0.98–1.01] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_alloc | large | 12 | 1.01 [1.00–1.01] | 1.01 [1.00–1.01] | 1.00 [0.99–1.00] | 1.00 |
| enetw | prox_g_out | short | 14 | 1.00 [0.97–1.03] | 1.01 [0.99–1.06] | 1.00 [0.96–1.05] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | short | 14 | 0.95 [0.87–1.03] | 0.94 [0.87–1.05] | 0.99 [0.97–1.00] | 1.00 |
| enetw | prox_in | medium | 14 | 0.84 [0.84–0.86] | 0.84 [0.84–0.86] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | large | 12 | 0.85 [0.84–0.86] | 0.85 [0.84–0.86] | 1.00 [0.98–1.00] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.96 [0.88–1.02] | 0.95 [0.88–1.01] | 1.00 [0.98–1.01] | 0.99 |
| enetw | prox_incopy | medium | 14 | 0.85 [0.84–0.87] | 0.86 [0.84–0.89] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.87 [0.85–0.94] | 0.87 [0.85–0.94] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_out | short | 14 | 1.00 [0.95–1.05] | 1.01 [0.95–1.08] | 0.99 [0.97–1.01] | 1.00 |
| enetw | prox_out | medium | 14 | 0.92 [0.90–0.94] | 0.92 [0.90–0.95] | 1.00 [0.99–1.00] | 1.00 |
| enetw | prox_out | large | 12 | 0.96 [0.91–1.00] | 0.96 [0.91–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | sparse_prox | large | 2 | 0.98 [0.98–0.98] | 0.98 [0.98–0.98] | 1.00 [1.00–1.00] | 1.00 |
| enetw | valprox_out | short | 14 | 1.00 [0.97–1.04] | 1.01 [0.97–1.07] | 1.00 [0.97–1.01] | 1.01 |
| enetw | valprox_out | medium | 14 | 0.95 [0.93–0.98] | 0.95 [0.93–0.99] | 1.00 [1.00–1.01] | 1.00 |
| enetw | valprox_out | large | 12 | 0.97 [0.94–1.00] | 0.97 [0.94–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 1.01 [0.93–1.08] | 1.02 [0.96–1.09] | 1.05 [0.95–1.10] | 1.05 |
| l1 | block_loop | medium | 4 | 0.79 [0.77–0.82] | 0.78 [0.77–0.80] | 0.78 [0.77–0.80] | 1.00 |
| l1 | block_one | short | 2 | 0.92 [0.92–0.92] | 0.92 [0.92–0.92] | 0.92 [0.92–0.92] | 1.00 |
| l1 | prox_alloc | short | 14 | 1.00 [0.99–1.03] | 1.03 [1.00–1.07] | 1.00 [0.99–1.02] | 1.00 |
| l1 | prox_alloc | medium | 14 | 1.01 [0.98–1.02] | 1.01 [0.98–1.03] | 1.01 [0.98–1.03] | 1.00 |
| l1 | prox_alloc | large | 12 | 1.02 [1.00–1.04] | 1.03 [1.01–1.05] | 1.03 [1.02–1.04] | 1.00 |
| l1 | prox_g_out | short | 14 | 1.00 [0.97–1.06] | 0.96 [0.89–1.04] | 0.97 [0.89–1.05] | 1.04 |
| l1 | prox_g_out | medium | 14 | 0.98 [0.97–1.00] | 0.95 [0.93–0.98] | 0.95 [0.93–0.99] | 1.08 |
| l1 | prox_g_out | large | 12 | 0.99 [0.97–1.00] | 0.95 [0.93–0.98] | 0.95 [0.93–0.98] | 1.07 |
| l1 | prox_in | short | 14 | 0.89 [0.76–1.05] | 0.88 [0.76–1.05] | 0.88 [0.76–1.05] | 1.00 |
| l1 | prox_in | medium | 14 | 0.78 [0.76–0.83] | 0.79 [0.76–0.83] | 0.79 [0.76–0.83] | 1.00 |
| l1 | prox_in | large | 12 | 0.78 [0.76–0.82] | 0.79 [0.77–0.82] | 0.79 [0.77–0.82] | 1.01 |
| l1 | prox_incopy | short | 14 | 0.91 [0.73–1.10] | 0.92 [0.74–1.17] | 0.90 [0.73–1.03] | 1.00 |
| l1 | prox_incopy | medium | 14 | 0.82 [0.76–0.86] | 0.82 [0.78–0.86] | 0.82 [0.77–0.86] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.84 [0.79–0.94] | 0.84 [0.79–0.94] | 0.84 [0.78–0.94] | 1.00 |
| l1 | prox_out | short | 14 | 0.97 [0.82–1.09] | 1.04 [0.85–1.19] | 0.96 [0.84–1.05] | 1.01 |
| l1 | prox_out | medium | 14 | 0.78 [0.75–0.81] | 0.79 [0.75–0.86] | 0.78 [0.75–0.81] | 1.00 |
| l1 | prox_out | large | 12 | 0.86 [0.76–1.01] | 0.86 [0.76–1.01] | 0.86 [0.76–1.01] | 1.00 |
| l1 | sparse_prox | large | 2 | 0.97 [0.96–0.99] | 0.97 [0.96–0.98] | 0.97 [0.96–0.98] | 1.01 |
| l1 | valprox_out | short | 14 | 0.98 [0.88–1.03] | 1.02 [0.90–1.11] | 0.98 [0.88–1.03] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.85 [0.82–0.90] | 0.85 [0.83–0.92] | 0.85 [0.82–0.90] | 1.00 |
| l1 | valprox_out | large | 12 | 0.90 [0.83–1.00] | 0.90 [0.83–1.00] | 0.90 [0.83–1.00] | 1.00 |
| l1w | block_loop | short | 4 | 0.99 [0.94–1.03] | 0.99 [0.95–1.02] | 0.98 [0.95–1.00] | 1.00 |
| l1w | block_loop | medium | 4 | 0.87 [0.86–0.88] | 0.87 [0.86–0.88] | 0.87 [0.86–0.89] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 1.00 [0.98–1.01] | 1.00 [0.98–1.02] | 1.00 [0.98–1.01] | 1.00 |
| l1w | prox_alloc | medium | 14 | 1.00 [0.98–1.01] | 1.00 [0.98–1.01] | 1.00 [0.98–1.01] | 1.00 |
| l1w | prox_alloc | large | 12 | 1.00 [0.99–1.02] | 1.00 [0.99–1.02] | 1.01 [1.00–1.01] | 1.00 |
| l1w | prox_g_out | short | 14 | 0.95 [0.88–1.03] | 0.95 [0.88–1.04] | 0.95 [0.88–1.04] | 1.03 |
| l1w | prox_g_out | medium | 14 | 0.89 [0.88–0.90] | 0.89 [0.88–0.90] | 0.89 [0.88–0.90] | 1.03 |
| l1w | prox_g_out | large | 12 | 0.90 [0.88–0.97] | 0.90 [0.88–0.97] | 0.90 [0.88–0.97] | 1.02 |
| l1w | prox_in | short | 14 | 0.92 [0.86–1.03] | 0.94 [0.86–1.03] | 0.92 [0.87–1.00] | 1.00 |
| l1w | prox_in | medium | 14 | 0.84 [0.83–0.85] | 0.84 [0.83–0.86] | 0.84 [0.83–0.85] | 1.00 |
| l1w | prox_in | large | 12 | 0.84 [0.83–0.87] | 0.86 [0.83–0.89] | 0.85 [0.83–0.88] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.92 [0.77–1.01] | 0.96 [0.87–1.08] | 0.92 [0.79–1.00] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.85 [0.83–0.87] | 0.85 [0.84–0.87] | 0.85 [0.83–0.86] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.87 [0.84–0.96] | 0.88 [0.85–0.96] | 0.87 [0.84–0.95] | 1.00 |
| l1w | prox_out | short | 14 | 0.94 [0.89–1.02] | 1.01 [0.91–1.10] | 0.95 [0.88–1.02] | 1.00 |
| l1w | prox_out | medium | 14 | 0.88 [0.85–0.92] | 0.89 [0.85–0.92] | 0.88 [0.85–0.92] | 1.00 |
| l1w | prox_out | large | 12 | 0.96 [0.90–1.01] | 0.96 [0.90–1.01] | 0.96 [0.90–1.01] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.00 [1.00–1.01] | 1.00 [0.99–1.00] | 1.00 [0.99–1.00] | 1.00 |
| l1w | valprox_out | short | 14 | 0.98 [0.93–1.00] | 1.02 [0.94–1.07] | 0.98 [0.93–1.00] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.93 [0.91–0.95] | 0.93 [0.91–0.95] | 0.93 [0.91–0.95] | 1.00 |
| l1w | valprox_out | large | 12 | 0.97 [0.94–1.00] | 0.97 [0.94–1.00] | 0.97 [0.94–1.00] | 1.00 |

A3M4: 0 of 1008 cells slower in every process, 253 faster in every process, 9 above 1.05 in some processes only, 203 undecided (control unclean)
  some    enet/prox_alloc/Float64/n262144/normal       1.000–1.125   250724.4 -> 253660.3 ns   output offsets [0, 0, 0, 0, 0, 0]
  some    l1/prox_in/Float64/n1/normal                 1.039–1.095   7.0 -> 7.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n1048576/normal      1.011–1.073   548814.5 -> 557212.5 ns   output offsets [64, 64, 64, 64, 128, 128]
  some    enet/prox_alloc/Float64/n16384/normal        1.022–1.073   12188.1 -> 12652.7 ns   output offsets [3328, 3968, 832, 960, 3008, 1536]
  some    enetw/prox_alloc/Float64/n262144/normal      0.979–1.065   328141.1 -> 330179.2 ns   output offsets [0, 0, 0, 0, 0, 0]
  some    l1/prox_alloc/Float64/n65536/normal          1.024–1.062   51653.1 -> 53823.5 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n32768/normal        1.009–1.058   11744.9 -> 12278.4 ns   output offsets [128, 64, 3584, 3840, 256, 1856]
  some    l1w/prox_alloc/Float64/n32/normal            0.988–1.056   52.5 -> 53.7 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    enet/prox_alloc/Float64/n4096/normal         1.008–1.054   3285.4 -> 3395.0 ns   output offsets [320, 3968, 3776, 1600, 1088, 3136]

A3dr: 3 of 1008 cells slower in every process, 247 faster in every process, 18 above 1.05 in some processes only, 196 undecided (control unclean)
  slower  l1/prox_alloc/Float64/n1/normal              1.056–1.082   26.3 -> 28.2 ns
  slower  l1w/valprox_out/Float32/n1/normal            1.053–1.076   14.9 -> 15.9 ns
  slower  l1w/valprox_out/Float64/n1/normal            1.051–1.077   14.9 -> 15.7 ns
  some    enetw/valprox_out/Float32/n64/normal         0.940–1.125   47.0 -> 48.3 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    l1w/prox_alloc/Float64/n16/normal            0.981–1.124   40.9 -> 40.8 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    l1w/prox_alloc/Float64/n8/normal             1.003–1.124   34.1 -> 34.7 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    enet/prox_alloc/Float64/n262144/normal       0.972–1.102   250724.4 -> 263462.2 ns   output offsets [0, 0, 0, 0, 0, 0]
  some    l1w/prox_out/Float64/n16/normal              0.962–1.088   12.9 -> 12.8 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    l1/valprox_out/Float64/n16/normal            0.963–1.085   20.5 -> 20.8 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    enetw/valprox_out/Float32/n128/normal        0.976–1.082   62.5 -> 61.9 ns   output offsets [544, 544, 544, 544, 544, 544]
  some    l1/prox_alloc/Float32/n1/normal              1.048–1.079   26.8 -> 28.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n65536/normal          1.014–1.077   51653.1 -> 53563.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float32/n4/normal              1.026–1.071   28.1 -> 29.2 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_in/Float64/n1/normal                 1.039–1.070   7.0 -> 7.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n1048576/normal      1.006–1.062   548814.5 -> 556531.0 ns   output offsets [64, 64, 64, 64, 128, 128]
  some    l1w/prox_alloc/Float32/n64/normal            0.970–1.062   53.3 -> 52.8 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    l1/prox_alloc/Float64/n2/normal              1.035–1.058   27.5 -> 28.9 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_alloc/Float32/n2/normal              1.022–1.057   28.1 -> 29.2 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n32768/normal        1.008–1.055   11744.9 -> 12329.4 ns   output offsets [128, 64, 3584, 3840, 256, 1856]
  some    enetw/prox_alloc/Float64/n262144/normal      0.982–1.052   328141.1 -> 331269.5 ns   output offsets [0, 0, 0, 0, 0, 0]
  some    l1w/valprox_out/Float32/n8/normal            1.004–1.052   21.6 -> 22.4 ns   output offsets [64, 64, 64, 64, 64, 64]

A3n: 0 of 1008 cells slower in every process, 113 faster in every process, 7 above 1.05 in some processes only, 131 undecided (control unclean)
  some    l1/prox_in/Float64/n1/normal                 1.048–1.095   7.0 -> 7.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/block_one/Float64/n1/normal               0.917–1.083   4.0 -> 3.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float64/n262144/normal        0.988–1.071   339355.8 -> 340806.5 ns   output offsets [0, 0, 0, 0, 0, 0]
  some    enetw/prox_incopy/Float64/n64/normal         0.982–1.065   125.1 -> 126.0 ns   output offsets [544, 544, 544, 544, 544, 544]
  some    enetw/prox_incopy/Float64/n32/normal         0.991–1.062   72.9 -> 73.8 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    enetw/prox_out/Float64/n256/normal           0.998–1.055   106.3 -> 107.6 ns   output offsets [3072, 3584, 128, 3456, 2112, 1984]
  some    enetw/prox_alloc/Float64/n262144/normal      0.945–1.052   328141.1 -> 325423.7 ns   output offsets [0, 0, 0, 0, 0, 0]
