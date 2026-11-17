
## epyc7513a-julia1.13.4 (3 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 4032 of 4032

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3d | A3dr | A3m | control |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 0.99 [0.95–1.00] | 0.98 [0.95–1.00] | 0.98 [0.95–1.00] | 0.98 [0.95–1.00] | 1.02 |
| enet | block_loop | medium | 4 | 0.74 [0.68–0.82] | 0.74 [0.68–0.82] | 0.73 [0.68–0.78] | 0.73 [0.69–0.81] | 1.00 |
| enet | block_one | short | 2 | 1.00 [1.00–1.00] | 0.98 [0.95–1.00] | 1.00 [1.00–1.00] | 0.98 [0.95–1.00] | 1.00 |
| enet | prox_alloc | short | 14 | 0.98 [0.87–1.00] | 0.98 [0.87–1.01] | 0.99 [0.88–1.03] | 0.99 [0.88–1.03] | 1.00 |
| enet | prox_alloc | medium | 14 | 0.92 [0.72–1.12] | 0.95 [0.89–1.12] | 0.95 [0.91–1.00] | 0.92 [0.87–0.99] | 0.97 |
| enet | prox_alloc | large | 12 | 0.98 [0.82–1.10] | 1.02 [0.78–1.21] | 1.00 [0.77–1.27] | 0.97 [0.87–1.16] | 1.05 |
| enet | prox_g_out | short | 14 | 1.00 [0.97–1.02] | 1.00 [0.97–1.03] | 0.99 [0.91–1.04] | 1.00 [0.96–1.02] | 0.99 |
| enet | prox_g_out | medium | 14 | 1.00 [0.99–1.00] | 1.00 [0.99–1.00] | 1.00 [1.00–1.00] | 1.00 [0.99–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 1.00 |
| enet | prox_in | short | 14 | 0.86 [0.71–1.00] | 0.88 [0.72–1.03] | 0.87 [0.72–1.03] | 0.88 [0.72–1.03] | 1.01 |
| enet | prox_in | medium | 14 | 0.70 [0.69–0.71] | 0.70 [0.69–0.71] | 0.70 [0.69–0.71] | 0.70 [0.69–0.71] | 1.00 |
| enet | prox_in | large | 12 | 0.69 [0.69–0.69] | 0.69 [0.69–0.69] | 0.69 [0.69–0.69] | 0.69 [0.69–0.69] | 1.00 |
| enet | prox_incopy | short | 14 | 0.88 [0.74–1.00] | 0.89 [0.74–1.02] | 0.89 [0.75–1.02] | 0.89 [0.74–1.02] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 |
| enet | prox_incopy | large | 12 | 0.73 [0.72–0.75] | 0.73 [0.72–0.75] | 0.73 [0.72–0.75] | 0.73 [0.72–0.75] | 1.00 |
| enet | prox_out | short | 14 | 0.95 [0.73–1.00] | 0.97 [0.74–1.03] | 0.99 [0.76–1.06] | 0.97 [0.74–1.03] | 1.01 |
| enet | prox_out | medium | 14 | 0.71 [0.68–0.81] | 0.77 [0.68–1.00] | 0.77 [0.68–1.00] | 0.71 [0.68–0.82] | 1.00 |
| enet | prox_out | large | 12 | 0.91 [0.70–1.34] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 0.91 [0.69–1.32] | 1.00 |
| enet | sparse_prox | large | 2 | 1.09 [1.08–1.10] | 1.10 [1.10–1.10] | 1.02 [0.95–1.09] | 1.08 [1.08–1.09] | 1.00 |
| enet | valprox_out | short | 14 | 0.98 [0.89–1.01] | 0.99 [0.90–1.02] | 1.00 [0.90–1.05] | 0.99 [0.90–1.03] | 1.00 |
| enet | valprox_out | medium | 14 | 0.79 [0.77–0.88] | 0.84 [0.76–1.00] | 0.84 [0.76–1.00] | 0.79 [0.76–0.89] | 1.00 |
| enet | valprox_out | large | 12 | 0.94 [0.78–1.23] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 0.94 [0.77–1.23] | 1.00 |
| enetw | block_loop | short | 4 | 0.99 [0.90–1.14] | 1.00 [0.93–1.14] | 1.01 [0.90–1.11] | 0.99 [0.91–1.11] | 0.99 |
| enetw | block_loop | medium | 4 | 0.88 [0.85–0.94] | 0.88 [0.85–0.93] | 0.94 [0.87–1.02] | 0.88 [0.86–0.89] | 1.00 |
| enetw | block_one | short | 2 | 0.98 [0.97–1.00] | 0.98 [0.97–1.00] | 0.98 [0.97–1.00] | 0.98 [0.97–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.98 [0.93–1.00] | 0.98 [0.93–1.01] | 1.00 [0.97–1.03] | 0.99 [0.94–1.01] | 1.00 |
| enetw | prox_alloc | medium | 14 | 0.95 [0.81–1.04] | 0.97 [0.91–1.01] | 0.97 [0.91–1.03] | 0.97 [0.91–1.06] | 0.99 |
| enetw | prox_alloc | large | 12 | 1.02 [1.00–1.04] | 0.99 [0.81–1.07] | 0.97 [0.77–1.15] | 1.00 [0.78–1.06] | 0.99 |
| enetw | prox_g_out | short | 14 | 1.01 [0.99–1.04] | 1.00 [0.99–1.03] | 1.00 [0.99–1.02] | 1.00 [0.99–1.02] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [0.99–1.01] | 1.00 [0.99–1.00] | 1.00 [0.99–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.01] | 1.00 [0.99–1.00] | 1.00 [0.99–1.01] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_in | short | 14 | 0.91 [0.81–1.00] | 0.92 [0.82–1.02] | 0.92 [0.82–1.02] | 0.92 [0.82–1.02] | 1.00 |
| enetw | prox_in | medium | 14 | 0.78 [0.78–0.80] | 0.78 [0.78–0.81] | 0.78 [0.78–0.80] | 0.79 [0.78–0.81] | 1.00 |
| enetw | prox_in | large | 12 | 0.79 [0.78–0.83] | 0.79 [0.78–0.83] | 0.79 [0.78–0.83] | 0.79 [0.78–0.83] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.92 [0.82–1.02] | 0.93 [0.83–1.02] | 0.93 [0.83–1.02] | 0.93 [0.83–1.02] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.80 [0.79–0.82] | 0.80 [0.79–0.81] | 0.80 [0.79–0.81] | 0.80 [0.79–0.82] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.81 [0.79–0.86] | 0.82 [0.80–0.85] | 0.81 [0.79–0.86] | 0.82 [0.80–0.86] | 1.00 |
| enetw | prox_out | short | 14 | 0.97 [0.86–1.02] | 0.98 [0.86–1.02] | 0.98 [0.85–1.05] | 0.99 [0.86–1.05] | 1.00 |
| enetw | prox_out | medium | 14 | 0.83 [0.74–0.97] | 0.84 [0.73–1.00] | 0.84 [0.73–1.00] | 0.83 [0.73–0.98] | 1.00 |
| enetw | prox_out | large | 12 | 1.03 [0.96–1.14] | 1.00 [0.98–1.00] | 1.00 [0.99–1.01] | 1.02 [0.97–1.13] | 1.00 |
| enetw | sparse_prox | large | 2 | 0.94 [0.92–0.96] | 0.99 [0.97–1.01] | 1.01 [0.95–1.07] | 1.01 [0.96–1.06] | 1.01 |
| enetw | valprox_out | short | 14 | 0.96 [0.83–1.02] | 0.97 [0.83–1.01] | 0.98 [0.84–1.03] | 0.98 [0.84–1.04] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.90 [0.83–0.99] | 0.90 [0.83–1.01] | 0.90 [0.83–1.01] | 0.90 [0.84–0.99] | 1.00 |
| enetw | valprox_out | large | 12 | 1.02 [0.98–1.10] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.01 [0.98–1.08] | 1.00 |
| l1 | block_loop | short | 4 | 1.01 [0.96–1.05] | 0.99 [0.96–1.05] | 0.99 [0.96–1.05] | 1.01 [0.96–1.05] | 1.00 |
| l1 | block_loop | medium | 4 | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 1.00 |
| l1 | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.98 [0.95–1.01] | 1.02 [0.98–1.03] | 1.02 [0.97–1.04] | 1.03 [0.98–1.18] | 0.99 |
| l1 | prox_alloc | medium | 14 | 0.93 [0.79–0.99] | 0.96 [0.92–1.01] | 0.95 [0.76–1.03] | 0.94 [0.74–1.02] | 0.97 |
| l1 | prox_alloc | large | 12 | 1.03 [0.96–1.10] | 1.01 [0.76–1.26] | 1.02 [0.95–1.17] | 1.03 [0.90–1.12] | 0.99 |
| l1 | prox_g_out | short | 14 | 1.00 [0.93–1.05] | 1.00 [0.93–1.05] | 1.00 [0.92–1.05] | 1.00 [0.92–1.05] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.99 [0.98–1.02] | 0.98 [0.97–0.99] | 0.98 [0.97–0.99] | 0.98 [0.97–0.99] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.98 [0.97–0.99] | 0.98 [0.97–0.99] | 0.98 [0.97–0.98] | 0.98 [0.97–0.98] | 1.00 |
| l1 | prox_in | short | 14 | 0.83 [0.68–1.00] | 0.83 [0.68–1.00] | 0.83 [0.68–1.00] | 0.84 [0.68–1.06] | 1.00 |
| l1 | prox_in | medium | 14 | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 1.00 |
| l1 | prox_in | large | 12 | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.86 [0.70–0.97] | 0.89 [0.72–1.06] | 0.89 [0.72–1.09] | 0.90 [0.72–1.09] | 1.00 |
| l1 | prox_incopy | medium | 14 | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 |
| l1 | prox_out | short | 14 | 0.95 [0.79–1.09] | 1.00 [0.79–1.09] | 1.01 [0.81–1.09] | 1.00 [0.79–1.11] | 1.00 |
| l1 | prox_out | medium | 14 | 0.68 [0.67–0.79] | 0.74 [0.67–1.00] | 0.74 [0.67–1.00] | 0.68 [0.67–0.81] | 1.00 |
| l1 | prox_out | large | 12 | 0.81 [0.67–1.02] | 1.00 [0.97–1.01] | 0.99 [0.95–1.00] | 0.84 [0.67–1.02] | 1.00 |
| l1 | sparse_prox | large | 2 | 0.98 [0.97–1.00] | 0.98 [0.97–0.99] | 0.98 [0.97–0.99] | 0.98 [0.97–1.00] | 0.99 |
| l1 | valprox_out | short | 14 | 0.96 [0.86–1.00] | 0.97 [0.87–1.03] | 1.00 [0.87–1.08] | 0.98 [0.86–1.03] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.77 [0.74–0.87] | 0.82 [0.74–1.00] | 0.82 [0.74–1.00] | 0.77 [0.74–0.88] | 1.00 |
| l1 | valprox_out | large | 12 | 0.86 [0.75–1.00] | 1.00 [0.99–1.05] | 0.99 [0.94–1.00] | 0.88 [0.75–1.02] | 1.00 |
| l1w | block_loop | short | 4 | 0.99 [0.93–1.04] | 0.96 [0.90–1.05] | 0.98 [0.90–1.09] | 0.97 [0.90–1.04] | 1.00 |
| l1w | block_loop | medium | 4 | 0.72 [0.68–0.76] | 0.72 [0.67–0.77] | 0.71 [0.68–0.76] | 0.71 [0.68–0.75] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 0.97 [0.90–1.00] | 1.00 [0.91–1.04] | 1.00 [0.95–1.10] | 0.99 [0.92–1.02] | 1.00 |
| l1w | prox_alloc | medium | 14 | 0.95 [0.71–1.09] | 0.98 [0.88–1.05] | 0.99 [0.88–1.13] | 0.98 [0.85–1.12] | 0.95 |
| l1w | prox_alloc | large | 12 | 1.02 [0.90–1.10] | 0.99 [0.82–1.08] | 0.98 [0.82–1.08] | 1.05 [0.96–1.13] | 0.97 |
| l1w | prox_g_out | short | 14 | 1.02 [0.91–1.09] | 0.99 [0.91–1.04] | 1.03 [0.95–1.10] | 1.02 [0.91–1.10] | 1.06 |
| l1w | prox_g_out | medium | 14 | 1.03 [0.96–1.11] | 0.98 [0.95–1.00] | 1.05 [1.00–1.11] | 1.03 [0.96–1.11] | 1.10 |
| l1w | prox_g_out | large | 12 | 1.03 [0.95–1.11] | 0.98 [0.96–1.00] | 1.04 [1.00–1.10] | 1.02 [0.96–1.10] | 1.09 |
| l1w | prox_in | short | 14 | 0.90 [0.76–1.00] | 0.90 [0.76–1.00] | 0.90 [0.76–1.00] | 0.90 [0.76–1.00] | 1.00 |
| l1w | prox_in | medium | 14 | 0.72 [0.71–0.73] | 0.72 [0.71–0.73] | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 |
| l1w | prox_in | large | 12 | 0.71 [0.71–0.71] | 0.71 [0.71–0.71] | 0.71 [0.71–0.71] | 0.71 [0.71–0.71] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.91 [0.77–1.03] | 0.93 [0.78–1.09] | 0.93 [0.78–1.09] | 0.94 [0.78–1.12] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.74 [0.72–0.76] | 0.74 [0.72–0.76] | 0.74 [0.72–0.76] | 0.74 [0.72–0.76] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 1.00 |
| l1w | prox_out | short | 14 | 0.96 [0.80–1.00] | 1.00 [0.82–1.05] | 0.99 [0.81–1.05] | 0.99 [0.80–1.08] | 0.99 |
| l1w | prox_out | medium | 14 | 0.79 [0.67–0.99] | 0.79 [0.67–1.00] | 0.79 [0.67–1.00] | 0.79 [0.68–0.99] | 1.00 |
| l1w | prox_out | large | 12 | 1.01 [0.99–1.05] | 1.00 [1.00–1.01] | 1.01 [1.00–1.03] | 1.01 [0.99–1.05] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.01 [1.00–1.01] | 1.01 [1.00–1.01] | 1.02 [1.01–1.02] | 1.02 [1.02–1.02] | 1.00 |
| l1w | valprox_out | short | 14 | 0.96 [0.88–1.00] | 0.98 [0.88–1.03] | 1.00 [0.90–1.06] | 1.00 [0.89–1.06] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.89 [0.82–1.00] | 0.89 [0.83–1.00] | 0.89 [0.83–1.00] | 0.89 [0.83–1.00] | 1.00 |
| l1w | valprox_out | large | 12 | 1.01 [0.99–1.03] | 1.00 [1.00–1.01] | 1.01 [1.00–1.03] | 1.01 [0.99–1.04] | 1.00 |

A3M4: 15 of 1008 cells slower in every process, 396 faster in every process, 8 above 1.05 in some processes only, 60 undecided (control unclean)
  slower  enet/prox_out/Float64/n65536/normal          1.315–1.348   15282.1 -> 20609.9 ns
  slower  enet/prox_out/Float64/n32768/normal          1.288–1.359   7146.6 -> 9239.9 ns
  slower  enet/prox_out/Float32/n1048576/normal        1.279–1.326   111374.6 -> 146991.4 ns
  slower  enet/prox_out/Float32/n262144/normal         1.273–1.327   27927.1 -> 36819.7 ns
  slower  enet/valprox_out/Float64/n65536/normal       1.201–1.244   21393.0 -> 26602.5 ns
  slower  enet/valprox_out/Float32/n1048576/normal     1.201–1.238   156553.2 -> 188739.7 ns
  slower  enet/valprox_out/Float32/n262144/normal      1.198–1.232   39222.8 -> 47387.8 ns
  slower  enet/valprox_out/Float64/n32768/normal       1.192–1.273   9669.1 -> 11691.8 ns
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 11.2 ns
  slower  enetw/prox_out/Float64/n32768/normal         1.092–1.141   8632.2 -> 9848.3 ns
  slower  enetw/prox_out/Float32/n1048576/normal       1.088–1.173   139489.0 -> 153035.6 ns
  slower  enetw/prox_out/Float32/n262144/normal        1.064–1.158   34509.9 -> 39665.6 ns
  slower  enetw/valprox_out/Float32/n262144/normal     1.060–1.095   57207.1 -> 60862.5 ns
  slower  enetw/valprox_out/Float32/n1048576/normal    1.055–1.088   230213.3 -> 249071.5 ns
  slower  l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns
  some    l1/prox_out/Float64/n32768/normal            0.894–1.147   8378.3 -> 7526.6 ns   output offsets [64, 64, 128]
  some    l1w/prox_alloc/Float64/n4096/normal          1.047–1.111   1622.4 -> 1601.6 ns   output offsets [1280, 2624, 1152]
  some    enetw/prox_alloc/Float64/n65536/normal       0.870–1.069   22616.0 -> 20273.6 ns   output offsets [128, 64, 128]
  some    enetw/prox_g_out/Float64/n32/normal          1.000–1.059   53.0 -> 54.8 ns   output offsets [576, 288, 288]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.5 ns   output offsets [64, 64, 64]
  some    l1w/valprox_out/Float64/n32768/normal        1.002–1.052   14744.0 -> 15244.5 ns   output offsets [64, 64, 128]
  some    enetw/prox_g_out/Float64/n1/normal           1.000–1.051   16.1 -> 16.9 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float64/n4/normal                1.000–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 64]

A3d: 7 of 1008 cells slower in every process, 352 faster in every process, 11 above 1.05 in some processes only, 40 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 11.2 ns
  slower  l1w/prox_incopy/Float32/n1/normal            1.088–1.091   9.3 -> 10.1 ns
  slower  l1w/prox_out/Float64/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_out/Float32/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1/block_loop/Float64/n4/normal              1.053–1.053   5.2 -> 5.5 ns
  slower  l1/prox_out/Float32/n4/normal                1.050–1.100   5.5 -> 5.7 ns
  slower  l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns
  some    enetw/prox_alloc/Float64/n65536/normal       0.994–1.221   22616.0 -> 22510.4 ns   output offsets [128, 64, 128]
  some    enet/valprox_out/Float64/n1/normal           0.991–1.136   15.6 -> 15.7 ns   output offsets [64, 64, 64]
  some    enetw/sparse_prox/Float32/n65536/tenth       0.958–1.076   20141.7 -> 19363.6 ns   output offsets [256, 3072, 3328]
  some    l1w/prox_alloc/Float64/n4096/normal          1.004–1.073   1622.4 -> 1575.8 ns   output offsets [1280, 2624, 1152]
  some    enetw/valprox_out/Float32/n1/normal          0.984–1.063   17.0 -> 17.0 ns   output offsets [64, 64, 64]
  some    l1/prox_alloc/Float32/n1/normal              1.014–1.062   17.5 -> 18.3 ns   output offsets [64, 64, 64]
  some    enetw/valprox_out/Float32/n16/normal         1.002–1.059   28.1 -> 28.7 ns   output offsets [96, 96, 96]
  some    enet/valprox_out/Float32/n8/normal           0.987–1.059   17.8 -> 18.3 ns   output offsets [64, 128, 64]
  some    enetw/prox_incopy/Float64/n1/normal          1.000–1.057   14.8 -> 15.0 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n1/normal                1.000–1.056   4.9 -> 4.9 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float64/n4/normal                1.000–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 64]

A3dr: 5 of 1008 cells slower in every process, 350 faster in every process, 22 above 1.05 in some processes only, 49 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 10.9 ns
  slower  l1w/prox_incopy/Float32/n1/normal            1.088–1.091   9.3 -> 10.1 ns
  slower  l1/prox_out/Float32/n1/normal                1.055–1.056   4.9 -> 5.2 ns
  slower  l1w/prox_out/Float32/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_out/Float64/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  some    enetw/prox_alloc/Float64/n65536/normal       0.994–1.235   22616.0 -> 23302.0 ns   output offsets [128, 64, 128]
  some    enet/prox_alloc/Float64/n32/normal           0.985–1.198   29.7 -> 35.3 ns   output offsets [576, 288, 288]
  some    l1w/prox_alloc/Float64/n4096/normal          0.983–1.138   1622.4 -> 1583.4 ns   output offsets [1280, 2624, 1152]
  some    l1/prox_alloc/Float64/n32/normal             0.991–1.109   23.7 -> 23.9 ns   output offsets [576, 288, 288]
  some    enetw/valprox_out/Float32/n8/normal          0.995–1.104   21.4 -> 21.5 ns   output offsets [64, 128, 64]
  some    l1/prox_out/Float64/n4/normal                1.050–1.100   5.5 -> 5.7 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.050–1.100   5.5 -> 5.7 ns   output offsets [48, 96, 48]
  some    enetw/sparse_prox/Float32/n65536/tenth       0.909–1.078   20141.7 -> 19197.9 ns   output offsets [256, 3072, 3328]
  some    l1/valprox_out/Float32/n16/normal            1.007–1.057   13.3 -> 13.4 ns   output offsets [96, 96, 96]
  some    enetw/prox_incopy/Float64/n1/normal          0.982–1.057   14.8 -> 15.0 ns   output offsets [64, 64, 64]
  some    l1w/valprox_out/Float32/n8/normal            1.020–1.055   13.4 -> 14.0 ns   output offsets [64, 128, 64]
  some    enet/valprox_out/Float32/n262144/normal      1.000–1.053   39222.8 -> 39222.0 ns   output offsets [64, 64, 64]
  some    enet/prox_out/Float32/n4/normal              1.026–1.053   10.4 -> 10.9 ns   output offsets [48, 96, 48]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.5 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n262144/normal           0.936–1.052   32198.8 -> 30862.6 ns   output offsets [64, 64, 64]
  some    l1/valprox_out/Float64/n4/normal             1.026–1.051   10.7 -> 10.9 ns   output offsets [64, 64, 64]
  some    l1/valprox_out/Float32/n1048576/normal       0.918–1.051   169555.8 -> 164314.6 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float32/n4/normal           1.005–1.051   16.4 -> 17.0 ns   output offsets [48, 96, 48]
  some    enet/valprox_out/Float64/n1/normal           1.020–1.051   15.6 -> 16.4 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n2/normal                1.050–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float64/n1/normal              1.000–1.050   5.5 -> 5.5 ns   output offsets [64, 64, 64]

A3m: 13 of 1008 cells slower in every process, 387 faster in every process, 20 above 1.05 in some processes only, 47 undecided (control unclean)
  slower  enet/prox_out/Float64/n65536/normal          1.294–1.337   15282.1 -> 19779.7 ns
  slower  enet/prox_out/Float64/n32768/normal          1.286–1.302   7146.6 -> 9233.7 ns
  slower  enet/prox_out/Float32/n262144/normal         1.268–1.345   27927.1 -> 35875.0 ns
  slower  enet/valprox_out/Float64/n65536/normal       1.203–1.219   21393.0 -> 25727.4 ns
  slower  enet/valprox_out/Float32/n262144/normal      1.195–1.241   39222.8 -> 48224.9 ns
  slower  enet/valprox_out/Float64/n32768/normal       1.189–1.237   9669.1 -> 11731.3 ns
  slower  enetw/prox_out/Float32/n262144/normal        1.115–1.150   34509.9 -> 39054.1 ns
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 10.9 ns
  slower  enetw/prox_out/Float64/n32768/normal         1.102–1.140   8632.2 -> 9827.4 ns
  slower  l1w/prox_incopy/Float32/n1/normal            1.088–1.280   9.3 -> 10.1 ns
  slower  l1w/prox_alloc/Float64/n4096/normal          1.085–1.116   1622.4 -> 1774.5 ns
  slower  l1w/prox_out/Float32/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_out/Float64/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  some    enetw/prox_out/Float32/n64/normal            0.937–2.696   17.2 -> 16.7 ns   output offsets [288, 576, 288]
  some    enetw/prox_alloc/Float64/n65536/normal       0.864–1.236   22616.0 -> 23032.8 ns   output offsets [128, 64, 128]
  some    l1/prox_out/Float64/n32768/normal            0.902–1.144   8378.3 -> 7581.4 ns   output offsets [64, 64, 128]
  some    enetw/prox_incopy/Float32/n2/normal          0.984–1.127   16.1 -> 16.8 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n1/normal                1.000–1.111   4.9 -> 5.5 ns   output offsets [64, 64, 64]
  some    enetw/valprox_out/Float32/n262144/normal     1.047–1.108   57207.1 -> 60718.4 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n2/normal                1.000–1.100   5.5 -> 6.0 ns   output offsets [64, 64, 64]
  some    enetw/prox_alloc/Float32/n16/normal          0.990–1.096   28.0 -> 28.0 ns   output offsets [96, 96, 96]
  some    l1w/prox_alloc/Float32/n1/normal             1.004–1.094   18.6 -> 19.1 ns   output offsets [64, 64, 64]
  some    enetw/valprox_out/Float32/n16/normal         1.011–1.067   28.1 -> 28.7 ns   output offsets [96, 96, 96]
  some    l1w/valprox_out/Float64/n32768/normal        1.009–1.065   14744.0 -> 15497.0 ns   output offsets [64, 64, 128]
  some    l1/prox_alloc/Float32/n1/normal              1.029–1.061   17.5 -> 18.3 ns   output offsets [64, 64, 64]
  some    l1/prox_alloc/Float32/n2/normal              1.024–1.059   18.1 -> 18.8 ns   output offsets [64, 64, 64]
  some    enetw/prox_incopy/Float64/n1/normal          1.018–1.057   14.8 -> 15.3 ns   output offsets [64, 64, 64]
  some    l1/prox_in/Float32/n1/normal                 1.000–1.056   4.9 -> 5.2 ns   output offsets [64, 64, 64]
  some    l1w/valprox_out/Float32/n8/normal            0.972–1.055   13.4 -> 14.0 ns   output offsets [64, 128, 64]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.5 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float64/n1/normal              1.000–1.050   5.5 -> 5.5 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float64/n4/normal                1.000–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 64]

## epyc7513b-julia1.13.4 (3 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 4032 of 4032

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3d | A3dr | A3m | control |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 0.99 [0.94–1.03] | 0.99 [0.94–1.03] | 0.99 [0.94–1.03] | 0.99 [0.94–1.03] | 1.00 |
| enet | block_loop | medium | 4 | 0.74 [0.68–0.82] | 0.74 [0.68–0.80] | 0.74 [0.68–0.80] | 0.74 [0.68–0.80] | 1.00 |
| enet | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 1.02 [1.00–1.05] | 1.00 |
| enet | prox_alloc | short | 14 | 0.98 [0.86–1.00] | 0.99 [0.87–1.01] | 0.99 [0.88–1.02] | 0.98 [0.87–1.01] | 1.00 |
| enet | prox_alloc | medium | 14 | 0.89 [0.69–1.00] | 0.95 [0.88–1.22] | 0.96 [0.87–1.19] | 0.96 [0.88–1.23] | 0.97 |
| enet | prox_alloc | large | 12 | 0.93 [0.79–1.05] | 1.00 [0.99–1.03] | 1.00 [0.96–1.03] | 0.95 [0.85–1.01] | 0.97 |
| enet | prox_g_out | short | 14 | 0.99 [0.97–1.01] | 1.00 [0.98–1.04] | 1.00 [0.94–1.09] | 0.99 [0.98–1.02] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [0.98–1.00] | 1.00 [0.99–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [0.99–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | short | 14 | 0.86 [0.71–1.00] | 0.87 [0.72–1.00] | 0.86 [0.71–1.00] | 0.87 [0.71–1.03] | 1.00 |
| enet | prox_in | medium | 14 | 0.70 [0.69–0.71] | 0.70 [0.69–0.71] | 0.70 [0.69–0.71] | 0.70 [0.69–0.71] | 1.00 |
| enet | prox_in | large | 12 | 0.69 [0.69–0.69] | 0.69 [0.69–0.69] | 0.69 [0.69–0.69] | 0.69 [0.69–0.69] | 1.00 |
| enet | prox_incopy | short | 14 | 0.88 [0.74–1.00] | 0.89 [0.73–1.02] | 0.88 [0.74–1.00] | 0.89 [0.74–1.02] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 |
| enet | prox_incopy | large | 12 | 0.73 [0.72–0.75] | 0.73 [0.72–0.75] | 0.73 [0.72–0.75] | 0.73 [0.72–0.75] | 1.00 |
| enet | prox_out | short | 14 | 0.94 [0.74–1.00] | 0.96 [0.75–1.00] | 0.98 [0.76–1.03] | 0.96 [0.75–1.03] | 1.00 |
| enet | prox_out | medium | 14 | 0.71 [0.68–0.79] | 0.77 [0.68–1.00] | 0.77 [0.68–1.00] | 0.71 [0.68–0.81] | 1.00 |
| enet | prox_out | large | 12 | 0.90 [0.70–1.32] | 1.00 [1.00–1.00] | 1.00 [1.00–1.02] | 0.90 [0.69–1.31] | 1.00 |
| enet | sparse_prox | large | 2 | 1.14 [1.11–1.16] | 1.13 [1.10–1.17] | 1.14 [1.10–1.17] | 1.13 [1.10–1.16] | 1.03 |
| enet | valprox_out | short | 14 | 0.97 [0.87–1.03] | 0.99 [0.91–1.04] | 1.00 [0.88–1.04] | 1.00 [0.89–1.05] | 1.00 |
| enet | valprox_out | medium | 14 | 0.79 [0.76–0.88] | 0.84 [0.76–1.00] | 0.84 [0.76–1.00] | 0.79 [0.76–0.91] | 1.00 |
| enet | valprox_out | large | 12 | 0.93 [0.78–1.23] | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 0.93 [0.77–1.23] | 1.00 |
| enetw | block_loop | short | 4 | 1.00 [0.93–1.11] | 1.01 [0.90–1.11] | 1.01 [0.93–1.14] | 1.00 [0.90–1.14] | 1.01 |
| enetw | block_loop | medium | 4 | 0.88 [0.86–0.90] | 1.00 [0.99–1.02] | 0.88 [0.85–0.93] | 0.87 [0.85–0.90] | 1.00 |
| enetw | block_one | short | 2 | 0.98 [0.97–1.00] | 0.99 [0.97–1.00] | 1.00 [1.00–1.00] | 0.98 [0.97–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.98 [0.94–1.01] | 1.01 [0.98–1.11] | 1.01 [0.97–1.12] | 1.00 [0.95–1.02] | 1.00 |
| enetw | prox_alloc | medium | 14 | 0.93 [0.86–1.02] | 0.97 [0.88–1.03] | 0.96 [0.88–1.08] | 0.97 [0.88–1.10] | 0.98 |
| enetw | prox_alloc | large | 12 | 0.99 [0.85–1.04] | 1.03 [0.97–1.24] | 1.04 [0.98–1.28] | 1.04 [0.90–1.31] | 1.05 |
| enetw | prox_g_out | short | 14 | 1.00 [0.98–1.01] | 1.00 [0.98–1.01] | 1.01 [0.99–1.02] | 1.00 [0.99–1.02] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | short | 14 | 0.92 [0.82–1.02] | 0.93 [0.82–1.05] | 0.92 [0.82–1.00] | 0.93 [0.82–1.02] | 1.00 |
| enetw | prox_in | medium | 14 | 0.78 [0.78–0.80] | 0.79 [0.78–0.82] | 0.78 [0.78–0.80] | 0.78 [0.78–0.80] | 1.00 |
| enetw | prox_in | large | 12 | 0.78 [0.78–0.82] | 0.78 [0.78–0.83] | 0.78 [0.78–0.83] | 0.78 [0.78–0.83] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.92 [0.83–1.02] | 0.94 [0.83–1.04] | 0.93 [0.83–1.07] | 0.93 [0.82–1.02] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.80 [0.79–0.81] | 0.80 [0.79–0.82] | 0.80 [0.79–0.82] | 0.80 [0.79–0.82] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.81 [0.80–0.84] | 0.81 [0.80–0.84] | 0.81 [0.80–0.84] | 0.81 [0.80–0.84] | 1.00 |
| enetw | prox_out | short | 14 | 0.97 [0.87–1.02] | 1.00 [0.89–1.05] | 1.00 [0.88–1.05] | 1.00 [0.88–1.05] | 1.00 |
| enetw | prox_out | medium | 14 | 0.83 [0.74–0.98] | 0.85 [0.74–1.01] | 0.85 [0.74–1.00] | 0.84 [0.74–0.98] | 1.00 |
| enetw | prox_out | large | 12 | 1.01 [0.95–1.11] | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 1.01 [0.95–1.10] | 1.00 |
| enetw | sparse_prox | large | 2 | 1.02 [0.97–1.06] | 1.02 [0.96–1.08] | 0.98 [0.96–1.00] | 1.07 [1.07–1.07] | 1.00 |
| enetw | valprox_out | short | 14 | 0.96 [0.87–1.01] | 0.98 [0.86–1.03] | 0.98 [0.87–1.03] | 0.98 [0.85–1.03] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.89 [0.84–0.99] | 0.90 [0.84–1.01] | 0.90 [0.84–1.00] | 0.90 [0.84–0.99] | 1.00 |
| enetw | valprox_out | large | 12 | 1.01 [0.97–1.08] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.01 [0.97–1.08] | 1.00 |
| l1 | block_loop | short | 4 | 0.97 [0.92–1.00] | 0.97 [0.92–1.00] | 0.97 [0.92–1.00] | 0.97 [0.92–1.00] | 1.00 |
| l1 | block_loop | medium | 4 | 0.69 [0.67–0.77] | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 1.00 |
| l1 | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.99 [0.96–1.02] | 1.02 [0.98–1.05] | 1.02 [0.98–1.04] | 1.02 [0.98–1.05] | 1.00 |
| l1 | prox_alloc | medium | 14 | 0.94 [0.82–1.02] | 0.97 [0.87–1.23] | 0.96 [0.88–1.08] | 0.95 [0.71–1.07] | 0.98 |
| l1 | prox_alloc | large | 12 | 1.01 [0.81–1.21] | 1.05 [0.98–1.32] | 1.03 [0.99–1.27] | 1.05 [0.86–1.33] | 1.01 |
| l1 | prox_g_out | short | 14 | 1.00 [0.93–1.05] | 1.01 [0.93–1.05] | 1.01 [0.93–1.05] | 1.01 [0.93–1.05] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.98 [0.97–1.00] | 0.98 [0.97–1.00] | 0.98 [0.97–1.00] | 0.98 [0.97–1.00] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.98 [0.96–0.98] | 0.98 [0.96–0.98] | 0.98 [0.96–0.99] | 0.98 [0.96–0.98] | 1.00 |
| l1 | prox_in | short | 14 | 0.83 [0.68–1.00] | 0.83 [0.68–1.00] | 0.83 [0.68–1.00] | 0.83 [0.68–1.00] | 1.00 |
| l1 | prox_in | medium | 14 | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 0.68 [0.68–0.69] | 1.00 |
| l1 | prox_in | large | 12 | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 0.68 [0.68–0.68] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.88 [0.71–1.03] | 0.90 [0.72–1.09] | 0.90 [0.72–1.09] | 0.90 [0.72–1.09] | 1.02 |
| l1 | prox_incopy | medium | 14 | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 0.71 [0.70–0.73] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 0.72 [0.71–0.74] | 1.00 |
| l1 | prox_out | short | 14 | 0.95 [0.77–1.05] | 0.99 [0.79–1.05] | 1.01 [0.79–1.10] | 0.98 [0.79–1.06] | 1.01 |
| l1 | prox_out | medium | 14 | 0.68 [0.67–0.77] | 0.74 [0.67–1.00] | 0.74 [0.67–1.00] | 0.68 [0.67–0.81] | 1.00 |
| l1 | prox_out | large | 12 | 0.80 [0.67–0.96] | 1.00 [0.98–1.01] | 1.01 [1.00–1.05] | 0.83 [0.67–1.01] | 1.00 |
| l1 | sparse_prox | large | 2 | 1.01 [1.00–1.02] | 1.02 [1.01–1.03] | 1.02 [1.01–1.03] | 0.99 [0.97–1.01] | 1.02 |
| l1 | valprox_out | short | 14 | 0.96 [0.86–1.01] | 0.98 [0.87–1.04] | 1.01 [0.87–1.08] | 0.98 [0.86–1.03] | 1.01 |
| l1 | valprox_out | medium | 14 | 0.77 [0.74–0.86] | 0.82 [0.74–1.01] | 0.82 [0.74–1.00] | 0.77 [0.74–0.88] | 1.00 |
| l1 | valprox_out | large | 12 | 0.85 [0.75–0.95] | 1.00 [0.99–1.01] | 1.00 [1.00–1.03] | 0.87 [0.76–1.00] | 1.00 |
| l1w | block_loop | short | 4 | 1.00 [0.93–1.09] | 0.99 [0.93–1.05] | 0.99 [0.93–1.05] | 0.99 [0.93–1.05] | 1.04 |
| l1w | block_loop | medium | 4 | 0.72 [0.68–0.77] | 0.72 [0.68–0.78] | 0.73 [0.68–0.79] | 0.72 [0.68–0.77] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.04 |
| l1w | prox_alloc | short | 14 | 0.98 [0.92–1.01] | 0.99 [0.94–1.03] | 0.99 [0.93–1.03] | 1.01 [0.95–1.13] | 1.00 |
| l1w | prox_alloc | medium | 14 | 0.93 [0.74–1.12] | 0.96 [0.87–1.07] | 0.98 [0.88–1.11] | 1.01 [0.88–1.16] | 0.99 |
| l1w | prox_alloc | large | 12 | 1.05 [0.92–1.10] | 1.00 [0.96–1.05] | 1.04 [0.99–1.29] | 1.06 [0.84–1.38] | 1.05 |
| l1w | prox_g_out | short | 14 | 0.99 [0.91–1.04] | 0.99 [0.91–1.04] | 0.96 [0.83–1.04] | 0.97 [0.83–1.04] | 0.97 |
| l1w | prox_g_out | medium | 14 | 0.98 [0.96–1.01] | 0.98 [0.96–1.00] | 0.94 [0.87–1.00] | 0.94 [0.87–1.00] | 0.96 |
| l1w | prox_g_out | large | 12 | 0.98 [0.96–1.00] | 0.98 [0.96–1.00] | 0.94 [0.88–1.00] | 0.94 [0.88–1.00] | 0.96 |
| l1w | prox_in | short | 14 | 0.90 [0.76–1.00] | 0.89 [0.77–1.00] | 0.89 [0.77–1.00] | 0.90 [0.76–1.00] | 1.00 |
| l1w | prox_in | medium | 14 | 0.72 [0.71–0.73] | 0.72 [0.71–0.73] | 0.72 [0.71–0.73] | 0.72 [0.71–0.73] | 1.00 |
| l1w | prox_in | large | 12 | 0.71 [0.71–0.71] | 0.71 [0.71–0.71] | 0.71 [0.71–0.71] | 0.71 [0.71–0.71] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.90 [0.77–1.00] | 0.93 [0.78–1.06] | 0.92 [0.78–1.06] | 0.92 [0.78–1.04] | 0.99 |
| l1w | prox_incopy | medium | 14 | 0.74 [0.72–0.75] | 0.74 [0.72–0.76] | 0.74 [0.72–0.75] | 0.74 [0.72–0.75] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 0.75 [0.73–0.76] | 1.00 |
| l1w | prox_out | short | 14 | 0.97 [0.80–1.09] | 0.99 [0.81–1.05] | 0.99 [0.82–1.05] | 1.00 [0.81–1.09] | 1.00 |
| l1w | prox_out | medium | 14 | 0.79 [0.67–1.00] | 0.79 [0.67–1.00] | 0.80 [0.67–1.00] | 0.79 [0.68–1.00] | 1.00 |
| l1w | prox_out | large | 12 | 1.01 [0.99–1.05] | 1.00 [1.00–1.01] | 1.00 [0.98–1.00] | 1.00 [0.99–1.04] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.01 [1.01–1.02] | 1.01 [1.00–1.01] | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 0.99 |
| l1w | valprox_out | short | 14 | 0.98 [0.89–1.04] | 0.98 [0.91–1.03] | 1.01 [0.92–1.08] | 0.99 [0.91–1.06] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.89 [0.83–1.00] | 0.89 [0.83–1.00] | 0.90 [0.83–1.00] | 0.89 [0.83–1.00] | 1.00 |
| l1w | valprox_out | large | 12 | 1.01 [0.99–1.03] | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 1.00 [0.99–1.03] | 1.00 |

A3M4: 8 of 1008 cells slower in every process, 409 faster in every process, 17 above 1.05 in some processes only, 57 undecided (control unclean)
  slower  enet/prox_out/Float32/n262144/normal         1.310–1.345   27929.3 -> 36836.8 ns
  slower  enet/prox_out/Float32/n1048576/normal        1.309–1.348   111905.9 -> 147429.9 ns
  slower  enet/valprox_out/Float32/n1048576/normal     1.225–1.258   156305.8 -> 193561.5 ns
  slower  enet/valprox_out/Float32/n262144/normal      1.223–1.257   39206.8 -> 48306.8 ns
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 10.9 ns
  slower  enetw/prox_out/Float32/n1048576/normal       1.076–1.141   139070.4 -> 152448.2 ns
  slower  enetw/valprox_out/Float32/n262144/normal     1.069–1.087   57429.7 -> 61946.9 ns
  slower  l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns
  some    enetw/prox_out/Float64/n32/normal            0.938–1.319   17.2 -> 16.4 ns   output offsets [288, 288, 288]
  some    enet/prox_out/Float64/n32768/normal          0.867–1.263   7195.8 -> 9040.6 ns   output offsets [64, 64, 64]
  some    enet/prox_out/Float64/n65536/normal          0.848–1.229   14658.3 -> 17892.9 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float64/n32768/normal       0.905–1.210   9538.1 -> 11499.2 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float64/n65536/normal       0.893–1.172   20652.8 -> 23835.4 ns   output offsets [64, 64, 64]
  some    enetw/prox_out/Float32/n262144/normal        1.036–1.131   34512.4 -> 38121.5 ns   output offsets [64, 64, 64]
  some    l1w/valprox_out/Float64/n8/normal            0.983–1.111   13.5 -> 13.4 ns   output offsets [1824, 96, 96]
  some    enetw/prox_alloc/Float32/n128/normal         0.901–1.107   42.7 -> 38.4 ns   output offsets [544, 544, 544]
  some    enetw/prox_out/Float64/n65536/normal         1.002–1.102   18437.5 -> 20109.6 ns   output offsets [64, 64, 64]
  some    enetw/prox_alloc/Float64/n64/normal          0.934–1.095   41.5 -> 38.9 ns   output offsets [544, 2176, 544]
  some    enetw/valprox_out/Float64/n32768/normal      1.013–1.087   14485.5 -> 15785.2 ns   output offsets [64, 64, 64]
  some    enetw/valprox_out/Float64/n65536/normal      1.004–1.068   29979.1 -> 31895.2 ns   output offsets [64, 64, 64]
  some    enetw/valprox_out/Float32/n1048576/normal    1.024–1.066   229508.2 -> 236762.1 ns   output offsets [64, 64, 64]
  some    l1/prox_alloc/Float64/n4096/normal           1.014–1.060   1470.6 -> 1508.9 ns   output offsets [3456, 960, 2048]
  some    l1/prox_alloc/Float32/n1048576/normal        1.038–1.055   136877.5 -> 158287.5 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float64/n1/normal              1.000–1.050   5.5 -> 5.5 ns   output offsets [64, 64, 64]
  some    enet/block_one/Float32/n1/normal             0.952–1.050   5.5 -> 5.5 ns   output offsets [64, 64, 64]

A3d: 5 of 1008 cells slower in every process, 353 faster in every process, 35 above 1.05 in some processes only, 32 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 10.9 ns
  slower  l1w/prox_out/Float64/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_out/Float32/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns
  slower  l1/prox_g_out/Float64/n1/normal              1.050–1.050   5.5 -> 5.7 ns
  some    enetw/prox_in/Float64/n16/normal             0.834–1.293   24.4 -> 20.5 ns   output offsets [160, 160, 160]
  some    enetw/block_loop/Float32/n128/normal         0.891–1.265   21.9 -> 21.9 ns   output offsets [544, 544, 544]
  some    l1w/valprox_out/Float64/n8/normal            0.983–1.128   13.5 -> 13.4 ns   output offsets [1824, 96, 96]
  some    enetw/prox_alloc/Float32/n64/normal          0.954–1.126   33.4 -> 32.4 ns   output offsets [288, 288, 288]
  some    enetw/prox_alloc/Float64/n64/normal          0.965–1.117   41.5 -> 46.2 ns   output offsets [544, 2176, 544]
  some    l1w/prox_g_out/Float32/n1024/normal          0.870–1.108   1273.3 -> 1226.7 ns   output offsets [0, 3072, 2176]
  some    l1w/prox_g_out/Float32/n256/normal           0.873–1.108   325.2 -> 313.6 ns   output offsets [1088, 2176, 1088]
  some    l1w/prox_g_out/Float32/n1000/normal          0.889–1.108   1244.3 -> 1321.0 ns   output offsets [1536, 640, 2368]
  some    enetw/prox_incopy/Float64/n8/normal          0.972–1.107   19.7 -> 19.6 ns   output offsets [1824, 96, 96]
  some    l1w/prox_g_out/Float32/n16384/normal         0.871–1.107   20459.7 -> 19695.7 ns   output offsets [3392, 3904, 384]
  some    l1w/prox_g_out/Float32/n8192/normal          0.871–1.106   10290.1 -> 9917.8 ns   output offsets [3584, 64, 2944]
  some    l1w/prox_g_out/Float32/n100000/normal        0.873–1.105   125248.4 -> 120737.0 ns   output offsets [2048, 2752, 2752]
  some    l1w/prox_g_out/Float32/n65536/normal         0.872–1.105   82257.8 -> 79117.4 ns   output offsets [3072, 1024, 1536]
  some    l1w/prox_g_out/Float32/n128/normal           0.885–1.104   167.6 -> 163.1 ns   output offsets [544, 544, 544]
  some    l1w/prox_g_out/Float32/n1048576/normal       0.873–1.104   1314890.5 -> 1266259.5 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float32/n262144/normal        0.873–1.104   328463.7 -> 316749.8 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float32/n32768/normal         0.873–1.104   41345.1 -> 39808.4 ns   output offsets [3584, 3904, 512]
  some    l1w/prox_g_out/Float32/n2048/normal          0.869–1.104   2549.9 -> 2450.2 ns   output offsets [1664, 3264, 1728]
  some    enetw/block_loop/Float32/n1024/normal        0.873–1.102   116.2 -> 119.5 ns   output offsets [0, 3072, 2176]
  some    l1w/prox_g_out/Float32/n32/normal            0.887–1.099   43.8 -> 42.7 ns   output offsets [160, 160, 160]
  some    l1w/prox_g_out/Float32/n4096/normal          0.873–1.098   5196.1 -> 5005.7 ns   output offsets [3136, 2752, 3072]
  some    enetw/valprox_out/Float64/n1/normal          1.016–1.091   17.8 -> 18.4 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float32/n16/normal            0.896–1.091   24.1 -> 23.5 ns   output offsets [288, 96, 96]
  some    l1w/prox_g_out/Float32/n8/normal             0.929–1.077   14.2 -> 14.2 ns   output offsets [64, 64, 64]
  some    enetw/sparse_prox/Float32/n65536/tenth       0.960–1.071   20251.7 -> 19521.5 ns   output offsets [3072, 1024, 1536]

A3dr: 10 of 1008 cells slower in every process, 356 faster in every process, 35 above 1.05 in some processes only, 42 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 11.2 ns
  slower  l1/prox_out/Float32/n1/normal                1.056–1.056   4.9 -> 5.2 ns
  slower  l1w/valprox_out/Float32/n1/normal            1.054–1.054   10.1 -> 10.7 ns
  slower  l1w/prox_out/Float32/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_out/Float64/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1/valprox_out/Float64/n4/normal             1.051–1.077   10.7 -> 11.2 ns
  slower  l1/prox_out/Float64/n4/normal                1.050–1.100   5.5 -> 6.0 ns
  slower  l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns
  slower  l1/prox_out/Float32/n4/normal                1.050–1.100   5.5 -> 5.7 ns
  slower  l1/prox_out/Float32/n2/normal                1.050–1.050   5.5 -> 5.7 ns
  some    enet/prox_alloc/Float32/n1/normal            1.013–1.181   22.7 -> 22.9 ns   output offsets [64, 64, 64]
  some    enet/prox_alloc/Float32/n16/normal           0.990–1.150   26.0 -> 25.8 ns   output offsets [288, 96, 96]
  some    enet/prox_alloc/Float32/n8/normal            0.998–1.136   24.9 -> 24.9 ns   output offsets [64, 64, 64]
  some    l1w/prox_alloc/Float32/n64/normal            0.935–1.135   28.2 -> 26.6 ns   output offsets [288, 288, 288]
  some    enetw/prox_alloc/Float64/n64/normal          0.930–1.124   41.5 -> 46.3 ns   output offsets [544, 2176, 544]
  some    enet/prox_alloc/Float32/n128/normal          0.930–1.120   37.4 -> 35.1 ns   output offsets [544, 544, 544]
  some    l1w/prox_g_out/Float32/n1024/normal          0.962–1.108   1273.3 -> 1409.9 ns   output offsets [0, 3072, 2176]
  some    l1w/prox_g_out/Float32/n256/normal           0.964–1.108   325.2 -> 359.7 ns   output offsets [1088, 2176, 1088]
  some    l1w/prox_g_out/Float32/n1000/normal          0.963–1.108   1244.3 -> 1377.2 ns   output offsets [1536, 640, 2368]
  some    l1w/prox_g_out/Float32/n16384/normal         0.962–1.106   20459.7 -> 22434.2 ns   output offsets [3392, 3904, 384]
  some    enet/prox_alloc/Float32/n2/normal            0.992–1.106   23.5 -> 24.0 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float32/n100000/normal        0.964–1.106   125248.4 -> 138126.6 ns   output offsets [2048, 2752, 2752]
  some    l1w/prox_g_out/Float32/n8192/normal          0.962–1.106   10290.1 -> 11329.6 ns   output offsets [3584, 64, 2944]
  some    l1w/prox_g_out/Float32/n65536/normal         0.963–1.105   82257.8 -> 90610.4 ns   output offsets [3072, 1024, 1536]
  some    l1w/prox_g_out/Float32/n128/normal           0.967–1.104   167.6 -> 184.7 ns   output offsets [544, 544, 544]
  some    l1w/prox_g_out/Float32/n32768/normal         0.964–1.104   41345.1 -> 45350.5 ns   output offsets [3584, 3904, 512]
  some    l1w/prox_g_out/Float32/n1048576/normal       0.963–1.104   1314890.5 -> 1447201.0 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float32/n2048/normal          0.962–1.104   2549.9 -> 2810.6 ns   output offsets [1664, 3264, 1728]
  some    l1w/prox_g_out/Float32/n262144/normal        0.964–1.103   328463.7 -> 362196.0 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float32/n32/normal            0.976–1.100   43.8 -> 48.1 ns   output offsets [160, 160, 160]
  some    enetw/prox_alloc/Float64/n8/normal           1.014–1.099   26.7 -> 27.2 ns   output offsets [1824, 96, 96]
  some    l1w/prox_g_out/Float32/n4096/normal          0.963–1.098   5196.1 -> 5630.3 ns   output offsets [3136, 2752, 3072]
  some    l1w/valprox_out/Float64/n8/normal            1.037–1.095   13.5 -> 14.2 ns   output offsets [1824, 96, 96]
  some    l1w/prox_g_out/Float32/n16/normal            0.977–1.091   24.1 -> 26.3 ns   output offsets [288, 96, 96]
  some    l1w/prox_out/Float64/n4/normal               1.000–1.087   6.3 -> 6.6 ns   output offsets [64, 64, 64]

A3m: 6 of 1008 cells slower in every process, 392 faster in every process, 41 above 1.05 in some processes only, 41 undecided (control unclean)
  slower  enet/prox_out/Float32/n262144/normal         1.283–1.316   27929.3 -> 36681.9 ns
  slower  enet/valprox_out/Float32/n262144/normal      1.217–1.238   39206.8 -> 48199.2 ns
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 11.2 ns
  slower  l1w/prox_out/Float64/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_out/Float32/n1/normal               1.053–1.053   5.2 -> 5.5 ns
  slower  l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns
  some    enet/prox_out/Float64/n32768/normal          0.848–1.314   7195.8 -> 8580.9 ns   output offsets [64, 64, 64]
  some    enet/prox_out/Float64/n65536/normal          0.847–1.290   14658.3 -> 17304.7 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float64/n32768/normal       0.885–1.239   9538.1 -> 11052.1 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float32/n1/normal           0.983–1.202   14.2 -> 14.2 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float64/n65536/normal       0.888–1.176   20652.8 -> 23680.8 ns   output offsets [64, 64, 64]
  some    enetw/prox_out/Float32/n262144/normal        1.048–1.142   34512.4 -> 37891.7 ns   output offsets [64, 64, 64]
  some    l1w/valprox_out/Float64/n8/normal            1.012–1.125   13.5 -> 13.7 ns   output offsets [1824, 96, 96]
  some    enetw/prox_alloc/Float64/n64/normal          0.922–1.115   41.5 -> 39.3 ns   output offsets [544, 2176, 544]
  some    enetw/prox_alloc/Float32/n128/normal         0.896–1.111   42.7 -> 47.0 ns   output offsets [544, 544, 544]
  some    l1w/prox_g_out/Float32/n1000/normal          0.964–1.108   1244.3 -> 1377.2 ns   output offsets [1536, 640, 2368]
  some    l1w/prox_g_out/Float32/n1024/normal          0.963–1.108   1273.3 -> 1409.8 ns   output offsets [0, 3072, 2176]
  some    l1w/prox_g_out/Float32/n256/normal           0.964–1.107   325.2 -> 359.7 ns   output offsets [1088, 2176, 1088]
  some    enet/valprox_out/Float32/n8/normal           1.019–1.106   17.7 -> 18.0 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float32/n2048/normal          0.961–1.106   2549.9 -> 2811.2 ns   output offsets [1664, 3264, 1728]
  some    l1w/prox_g_out/Float32/n128/normal           0.964–1.105   167.6 -> 184.7 ns   output offsets [544, 544, 544]
  some    l1w/prox_g_out/Float32/n65536/normal         0.964–1.104   82257.8 -> 90616.3 ns   output offsets [3072, 1024, 1536]
  some    l1w/prox_g_out/Float32/n100000/normal        0.965–1.104   125248.4 -> 138273.9 ns   output offsets [2048, 2752, 2752]
  some    l1w/prox_g_out/Float32/n32768/normal         0.963–1.104   41345.1 -> 45351.0 ns   output offsets [3584, 3904, 512]
  some    l1w/prox_g_out/Float32/n8192/normal          0.964–1.104   10290.1 -> 11330.6 ns   output offsets [3584, 64, 2944]
  some    l1w/prox_g_out/Float32/n262144/normal        0.963–1.103   328463.7 -> 362172.7 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float32/n4096/normal          0.964–1.103   5196.1 -> 5631.9 ns   output offsets [3136, 2752, 3072]
  some    l1w/prox_g_out/Float32/n16384/normal         0.963–1.103   20459.7 -> 22434.9 ns   output offsets [3392, 3904, 384]
  some    l1w/prox_g_out/Float32/n1048576/normal       0.963–1.103   1314890.5 -> 1447352.0 ns   output offsets [64, 64, 64]
  some    l1w/prox_g_out/Float32/n32/normal            0.975–1.100   43.8 -> 48.1 ns   output offsets [160, 160, 160]
  some    enetw/prox_out/Float64/n65536/normal         1.000–1.098   18437.5 -> 19638.2 ns   output offsets [64, 64, 64]

## xeon2640v4-julia1.13.4 (6 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 4032 of 4032

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3d | A3dr | A3m | control |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 0.99 [0.95–1.03] | 0.99 [0.95–1.02] | 0.99 [0.95–1.03] | 0.99 [0.95–1.03] | 0.99 |
| enet | block_loop | medium | 4 | 0.70 [0.67–0.74] | 0.70 [0.67–0.74] | 0.70 [0.67–0.74] | 0.70 [0.67–0.74] | 1.00 |
| enet | block_one | short | 2 | 1.04 [1.02–1.06] | 1.04 [1.02–1.06] | 1.04 [1.02–1.06] | 1.04 [1.02–1.06] | 1.00 |
| enet | prox_alloc | short | 14 | 0.99 [0.91–1.02] | 0.99 [0.93–1.03] | 1.00 [0.94–1.04] | 0.99 [0.93–1.03] | 1.00 |
| enet | prox_alloc | medium | 14 | 0.97 [0.92–1.01] | 0.97 [0.92–1.00] | 0.97 [0.92–0.99] | 0.97 [0.90–1.01] | 1.00 |
| enet | prox_alloc | large | 12 | 1.00 [0.99–1.00] | 1.00 [0.99–1.01] | 1.00 [0.97–1.01] | 1.00 [0.97–1.00] | 1.00 |
| enet | prox_g_out | short | 14 | 1.00 [1.00–1.01] | 1.00 [0.99–1.02] | 1.00 [1.00–1.02] | 1.00 [1.00–1.03] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | short | 14 | 0.86 [0.69–1.02] | 0.88 [0.69–1.08] | 0.87 [0.71–1.05] | 0.88 [0.71–1.07] | 1.00 |
| enet | prox_in | medium | 14 | 0.67 [0.67–0.69] | 0.67 [0.67–0.69] | 0.67 [0.67–0.69] | 0.67 [0.67–0.69] | 1.00 |
| enet | prox_in | large | 12 | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 1.00 |
| enet | prox_incopy | short | 14 | 0.87 [0.72–1.01] | 0.89 [0.72–1.06] | 0.88 [0.71–1.05] | 0.89 [0.71–1.05] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 1.00 |
| enet | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 1.00 |
| enet | prox_out | short | 14 | 0.95 [0.74–1.02] | 1.00 [0.76–1.08] | 1.01 [0.77–1.10] | 0.99 [0.76–1.08] | 1.00 |
| enet | prox_out | medium | 14 | 0.69 [0.68–0.74] | 0.75 [0.68–1.00] | 0.69 [0.67–0.76] | 0.69 [0.68–0.75] | 1.00 |
| enet | prox_out | large | 12 | 0.89 [0.68–0.99] | 1.00 [1.00–1.00] | 0.89 [0.68–0.99] | 0.90 [0.68–1.00] | 1.00 |
| enet | sparse_prox | large | 2 | 0.93 [0.93–0.94] | 0.93 [0.93–0.94] | 0.93 [0.93–0.94] | 0.94 [0.93–0.94] | 1.00 |
| enet | valprox_out | short | 14 | 0.98 [0.82–1.02] | 1.00 [0.83–1.06] | 1.01 [0.84–1.07] | 1.00 [0.83–1.05] | 1.00 |
| enet | valprox_out | medium | 14 | 0.76 [0.74–0.82] | 0.81 [0.74–1.00] | 0.77 [0.74–0.84] | 0.76 [0.74–0.85] | 1.00 |
| enet | valprox_out | large | 12 | 0.92 [0.75–0.99] | 1.00 [1.00–1.00] | 0.92 [0.75–0.99] | 0.92 [0.75–1.00] | 1.00 |
| enetw | block_loop | short | 4 | 1.02 [0.90–1.08] | 1.03 [0.91–1.09] | 1.02 [0.90–1.08] | 1.02 [0.91–1.08] | 1.00 |
| enetw | block_loop | medium | 4 | 0.85 [0.79–0.97] | 0.82 [0.78–0.87] | 0.84 [0.79–0.91] | 0.82 [0.78–0.87] | 1.00 |
| enetw | block_one | short | 2 | 1.01 [1.00–1.01] | 1.00 [1.00–1.00] | 1.01 [1.00–1.03] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.98 [0.92–1.02] | 0.99 [0.92–1.04] | 0.99 [0.93–1.03] | 0.98 [0.93–1.01] | 1.00 |
| enetw | prox_alloc | medium | 14 | 0.97 [0.92–1.00] | 0.97 [0.92–1.01] | 0.97 [0.92–1.00] | 0.97 [0.92–1.00] | 1.00 |
| enetw | prox_alloc | large | 12 | 1.00 [0.99–1.00] | 1.00 [1.00–1.01] | 1.00 [0.99–1.00] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_g_out | short | 14 | 1.00 [0.99–1.02] | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 1.00 [0.99–1.00] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | short | 14 | 0.86 [0.71–1.00] | 0.87 [0.71–1.02] | 0.86 [0.71–1.01] | 0.87 [0.71–1.04] | 1.00 |
| enetw | prox_in | medium | 14 | 0.68 [0.67–0.70] | 0.68 [0.67–0.70] | 0.68 [0.67–0.70] | 0.68 [0.67–0.70] | 1.00 |
| enetw | prox_in | large | 12 | 0.67 [0.67–0.68] | 0.67 [0.67–0.69] | 0.67 [0.67–0.68] | 0.67 [0.67–0.68] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.88 [0.73–1.01] | 0.88 [0.73–1.00] | 0.88 [0.73–1.01] | 0.88 [0.73–1.01] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 1.00 |
| enetw | prox_out | short | 14 | 0.95 [0.81–1.00] | 0.96 [0.80–1.02] | 0.98 [0.82–1.04] | 0.97 [0.80–1.04] | 1.00 |
| enetw | prox_out | medium | 14 | 0.74 [0.68–0.82] | 0.78 [0.68–1.00] | 0.75 [0.68–0.82] | 0.74 [0.68–0.82] | 1.00 |
| enetw | prox_out | large | 12 | 0.97 [0.80–1.00] | 1.00 [1.00–1.00] | 0.97 [0.80–1.00] | 0.97 [0.80–1.00] | 1.00 |
| enetw | sparse_prox | large | 2 | 0.94 [0.93–0.96] | 0.94 [0.93–0.96] | 0.94 [0.93–0.96] | 0.94 [0.93–0.96] | 1.00 |
| enetw | valprox_out | short | 14 | 0.96 [0.86–1.00] | 0.96 [0.87–1.01] | 0.98 [0.87–1.04] | 0.97 [0.87–1.03] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.81 [0.78–0.85] | 0.84 [0.78–1.01] | 0.81 [0.78–0.85] | 0.81 [0.78–0.85] | 1.00 |
| enetw | valprox_out | large | 12 | 0.98 [0.85–1.00] | 1.00 [1.00–1.00] | 0.98 [0.85–1.00] | 0.98 [0.85–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 0.99 [0.94–1.04] | 0.99 [0.94–1.04] | 0.99 [0.94–1.04] | 0.99 [0.94–1.04] | 1.00 |
| l1 | block_loop | medium | 4 | 0.66 [0.64–0.68] | 0.66 [0.64–0.68] | 0.66 [0.64–0.68] | 0.66 [0.64–0.68] | 1.00 |
| l1 | block_one | short | 2 | 1.04 [1.00–1.07] | 1.07 [1.07–1.07] | 1.04 [1.00–1.07] | 1.07 [1.07–1.07] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.98 [0.93–1.05] | 1.00 [0.94–1.05] | 1.02 [0.94–1.08] | 1.00 [0.94–1.07] | 1.00 |
| l1 | prox_alloc | medium | 14 | 0.98 [0.93–1.00] | 0.98 [0.93–1.00] | 0.97 [0.93–1.00] | 0.97 [0.94–0.99] | 1.00 |
| l1 | prox_alloc | large | 12 | 1.00 [0.98–1.01] | 1.00 [0.99–1.02] | 1.00 [0.99–1.00] | 1.00 [0.99–1.02] | 1.00 |
| l1 | prox_g_out | short | 14 | 0.89 [0.80–1.04] | 0.90 [0.80–1.04] | 0.91 [0.80–1.04] | 0.90 [0.80–1.04] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.78 [0.78–0.79] | 0.78 [0.78–0.79] | 0.78 [0.78–0.79] | 0.78 [0.78–0.79] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.78 [0.78–0.80] | 0.78 [0.78–0.80] | 0.78 [0.78–0.80] | 0.78 [0.78–0.80] | 1.00 |
| l1 | prox_in | short | 14 | 0.83 [0.66–1.13] | 0.84 [0.66–1.13] | 0.83 [0.66–1.13] | 0.83 [0.66–1.13] | 0.99 |
| l1 | prox_in | medium | 14 | 0.67 [0.67–0.67] | 0.67 [0.66–0.67] | 0.67 [0.66–0.67] | 0.67 [0.67–0.67] | 1.00 |
| l1 | prox_in | large | 12 | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.84 [0.69–1.08] | 0.86 [0.70–1.11] | 0.86 [0.70–1.12] | 0.86 [0.70–1.12] | 1.00 |
| l1 | prox_incopy | medium | 14 | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 0.72 [0.69–0.75] | 1.00 |
| l1 | prox_out | short | 14 | 0.95 [0.65–1.13] | 0.96 [0.65–1.13] | 1.02 [0.66–1.25] | 0.97 [0.65–1.17] | 1.00 |
| l1 | prox_out | medium | 14 | 0.67 [0.65–0.68] | 0.73 [0.65–1.00] | 0.67 [0.66–0.67] | 0.67 [0.65–0.67] | 1.00 |
| l1 | prox_out | large | 12 | 0.88 [0.67–0.99] | 1.00 [1.00–1.00] | 0.89 [0.67–0.99] | 0.89 [0.67–1.00] | 1.00 |
| l1 | sparse_prox | large | 2 | 0.95 [0.95–0.96] | 0.95 [0.94–0.96] | 0.95 [0.95–0.95] | 0.95 [0.95–0.96] | 0.99 |
| l1 | valprox_out | short | 14 | 0.96 [0.75–1.05] | 0.95 [0.76–1.05] | 0.99 [0.76–1.10] | 0.96 [0.76–1.05] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.74 [0.73–0.76] | 0.79 [0.72–1.00] | 0.74 [0.73–0.77] | 0.74 [0.73–0.76] | 1.00 |
| l1 | valprox_out | large | 12 | 0.91 [0.75–0.99] | 1.00 [1.00–1.00] | 0.91 [0.75–0.99] | 0.92 [0.75–1.00] | 1.00 |
| l1w | block_loop | short | 4 | 0.97 [0.91–1.03] | 0.97 [0.91–1.03] | 0.98 [0.91–1.03] | 0.97 [0.91–1.03] | 1.00 |
| l1w | block_loop | medium | 4 | 0.68 [0.67–0.69] | 0.68 [0.67–0.69] | 0.68 [0.67–0.69] | 0.68 [0.67–0.69] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 0.98 [0.92–1.01] | 0.98 [0.92–1.02] | 0.99 [0.93–1.02] | 0.98 [0.92–1.02] | 1.00 |
| l1w | prox_alloc | medium | 14 | 0.98 [0.93–1.00] | 0.98 [0.93–1.00] | 0.98 [0.92–0.99] | 0.97 [0.93–0.99] | 1.00 |
| l1w | prox_alloc | large | 12 | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 1.00 |
| l1w | prox_g_out | short | 14 | 0.91 [0.84–1.04] | 0.91 [0.85–1.04] | 0.91 [0.84–1.04] | 0.91 [0.84–1.04] | 1.00 |
| l1w | prox_g_out | medium | 14 | 0.83 [0.83–0.84] | 0.83 [0.83–0.84] | 0.83 [0.83–0.84] | 0.83 [0.83–0.84] | 1.00 |
| l1w | prox_g_out | large | 12 | 0.84 [0.83–0.86] | 0.84 [0.83–0.86] | 0.84 [0.83–0.86] | 0.84 [0.83–0.86] | 1.00 |
| l1w | prox_in | short | 14 | 0.82 [0.68–1.04] | 0.82 [0.68–1.04] | 0.82 [0.68–1.04] | 0.82 [0.68–1.04] | 1.00 |
| l1w | prox_in | medium | 14 | 0.67 [0.67–0.68] | 0.67 [0.67–0.68] | 0.67 [0.67–0.68] | 0.67 [0.67–0.68] | 1.00 |
| l1w | prox_in | large | 12 | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 0.67 [0.67–0.67] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.84 [0.70–1.02] | 0.86 [0.70–1.06] | 0.86 [0.70–1.07] | 0.86 [0.70–1.07] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 0.69 [0.68–0.72] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.72 [0.69–0.80] | 0.72 [0.69–0.80] | 0.72 [0.69–0.80] | 0.72 [0.69–0.80] | 1.00 |
| l1w | prox_out | short | 14 | 0.93 [0.70–1.04] | 0.96 [0.71–1.10] | 1.02 [0.72–1.20] | 0.97 [0.71–1.10] | 1.00 |
| l1w | prox_out | medium | 14 | 0.72 [0.67–0.80] | 0.76 [0.67–1.00] | 0.72 [0.67–0.81] | 0.72 [0.67–0.81] | 1.00 |
| l1w | prox_out | large | 12 | 0.97 [0.79–1.00] | 1.00 [1.00–1.00] | 0.97 [0.79–1.00] | 0.97 [0.79–1.00] | 1.00 |
| l1w | sparse_prox | large | 2 | 0.95 [0.94–0.96] | 0.96 [0.95–0.96] | 0.95 [0.95–0.96] | 0.95 [0.94–0.96] | 1.00 |
| l1w | valprox_out | short | 14 | 0.96 [0.84–1.02] | 0.98 [0.84–1.05] | 1.00 [0.84–1.07] | 0.98 [0.84–1.06] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.83 [0.81–0.87] | 0.86 [0.81–1.00] | 0.83 [0.81–0.87] | 0.83 [0.81–0.87] | 1.00 |
| l1w | valprox_out | large | 12 | 0.98 [0.87–1.00] | 1.00 [1.00–1.00] | 0.98 [0.87–1.00] | 0.98 [0.87–1.00] | 1.00 |

A3M4: 3 of 1008 cells slower in every process, 474 faster in every process, 6 above 1.05 in some processes only, 67 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.125–1.126   7.1 -> 8.0 ns
  slower  l1/prox_in/Float32/n1/normal                 1.125–1.125   7.1 -> 8.0 ns
  slower  l1/block_one/Float32/n1/normal               1.071–1.071   4.1 -> 4.4 ns
  some    enet/prox_g_out/Float64/n4/normal            0.990–1.117   32.6 -> 32.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_alloc/Float32/n8/normal           0.994–1.082   40.1 -> 41.2 ns   output offsets [64, 64, 64, 256, 64, 64]
  some    l1/prox_incopy/Float64/n1/normal             1.047–1.079   11.2 -> 12.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/block_one/Float64/n1/normal               1.000–1.071   4.1 -> 4.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.034–1.069   8.6 -> 9.1 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_alloc/Float32/n4/normal            0.996–1.069   34.9 -> 35.6 ns   output offsets [48, 48, 48, 48, 48, 48]

A3d: 7 of 1008 cells slower in every process, 421 faster in every process, 15 above 1.05 in some processes only, 59 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.125–1.167   7.1 -> 8.0 ns
  slower  l1/prox_in/Float32/n1/normal                 1.125–1.125   7.1 -> 8.0 ns
  slower  l1/prox_incopy/Float32/n1/normal             1.107–1.136   13.0 -> 14.5 ns
  slower  l1/block_one/Float32/n1/normal               1.071–1.071   4.1 -> 4.4 ns
  slower  l1/prox_out/Float32/n4/normal                1.069–1.103   8.6 -> 9.1 ns
  slower  enet/prox_in/Float32/n1/normal               1.058–1.098   15.3 -> 16.4 ns
  slower  enet/prox_out/Float32/n1/normal              1.058–1.098   15.2 -> 16.4 ns
  some    enetw/prox_incopy/Float64/n4/normal          0.944–1.322   28.9 -> 27.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_out/Float64/n32/normal            0.885–1.275   29.8 -> 26.7 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    enet/prox_out/Float64/n1/normal              1.034–1.088   17.1 -> 18.3 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_in/Float64/n1/normal               1.034–1.088   17.1 -> 18.3 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float32/n2/normal              0.998–1.084   26.1 -> 26.6 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    l1/prox_incopy/Float32/n2/normal             1.045–1.081   14.1 -> 15.0 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    l1/prox_incopy/Float64/n1/normal             1.038–1.079   11.2 -> 12.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_g_out/Float32/n8/normal            0.997–1.074   29.3 -> 30.0 ns   output offsets [64, 64, 64, 256, 64, 64]
  some    l1/prox_alloc/Float32/n8/normal              1.015–1.074   27.8 -> 28.5 ns   output offsets [64, 64, 64, 256, 64, 64]
  some    l1/block_one/Float64/n1/normal               1.000–1.071   4.1 -> 4.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_incopy/Float32/n1/normal           1.041–1.070   20.9 -> 22.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float64/n16384/normal         0.993–1.061   16341.5 -> 16175.2 ns   output offsets [1600, 1856, 832, 320, 1792, 832]
  some    l1/prox_alloc/Float64/n4/normal              1.027–1.061   26.7 -> 27.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_out/Float32/n8/normal               1.028–1.056   10.6 -> 11.1 ns   output offsets [64, 64, 64, 256, 64, 64]
  some    enet/prox_out/Float32/n8/normal              1.029–1.050   18.0 -> 18.6 ns   output offsets [64, 64, 64, 256, 64, 64]

A3dr: 8 of 1008 cells slower in every process, 465 faster in every process, 19 above 1.05 in some processes only, 74 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.208–1.250   7.1 -> 8.6 ns
  slower  l1/prox_in/Float32/n1/normal                 1.125–1.125   7.1 -> 8.0 ns
  slower  l1/prox_incopy/Float32/n1/normal             1.107–1.136   13.0 -> 14.6 ns
  slower  l1/prox_out/Float32/n4/normal                1.103–1.172   8.6 -> 9.7 ns
  slower  l1w/prox_out/Float32/n8/normal               1.083–1.111   10.6 -> 11.7 ns
  slower  enet/prox_out/Float32/n1/normal              1.077–1.118   15.2 -> 16.5 ns
  slower  l1/block_one/Float32/n1/normal               1.071–1.071   4.1 -> 4.4 ns
  slower  l1/prox_alloc/Float32/n4/normal              1.055–1.126   26.5 -> 28.2 ns
  some    enetw/prox_incopy/Float64/n4/normal          0.948–1.266   28.9 -> 27.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float64/n1/normal              1.017–1.123   17.1 -> 18.9 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float32/n8/normal              1.042–1.114   27.8 -> 29.3 ns   output offsets [64, 64, 64, 256, 64, 64]
  some    l1/prox_alloc/Float32/n2/normal              1.036–1.089   26.1 -> 27.3 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    enet/prox_in/Float64/n1/normal               0.983–1.088   17.1 -> 18.3 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/valprox_out/Float64/n2/normal            0.997–1.086   17.7 -> 18.1 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_incopy/Float32/n2/normal             1.043–1.085   14.1 -> 15.0 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    enet/prox_out/Float32/n8/normal              1.033–1.083   18.0 -> 18.9 ns   output offsets [64, 64, 64, 256, 64, 64]
  some    enet/prox_out/Float64/n2/normal              1.000–1.081   18.6 -> 19.8 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_incopy/Float64/n1/normal             1.047–1.079   11.2 -> 12.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n4/normal              1.048–1.078   26.7 -> 28.3 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/block_one/Float64/n1/normal               1.000–1.071   4.1 -> 4.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_out/Float32/n1/normal             1.016–1.067   17.8 -> 18.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n1/normal            1.007–1.065   33.1 -> 34.2 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n8/normal              1.027–1.061   29.5 -> 30.9 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    l1w/prox_alloc/Float32/n8/normal             0.998–1.060   29.9 -> 30.2 ns   output offsets [64, 64, 64, 256, 64, 64]
  some    l1w/prox_alloc/Float32/n4/normal             1.004–1.053   28.1 -> 28.9 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1w/prox_alloc/Float32/n16/normal            0.978–1.053   33.6 -> 33.6 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    l1w/prox_alloc/Float32/n1/normal             0.998–1.051   26.1 -> 26.7 ns   output offsets [64, 64, 64, 64, 64, 64]

A3m: 4 of 1008 cells slower in every process, 467 faster in every process, 20 above 1.05 in some processes only, 62 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.125–1.208   7.1 -> 8.1 ns
  slower  l1/prox_in/Float32/n1/normal                 1.125–1.125   7.1 -> 8.0 ns
  slower  l1/prox_incopy/Float32/n1/normal             1.114–1.136   13.0 -> 14.6 ns
  slower  l1/block_one/Float32/n1/normal               1.071–1.071   4.1 -> 4.4 ns
  some    l1/prox_out/Float32/n4/normal                1.034–1.138   8.6 -> 9.1 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_in/Float32/n1/normal               1.038–1.098   15.3 -> 16.2 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float32/n1/normal              1.038–1.098   15.2 -> 16.2 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_incopy/Float32/n2/normal             1.049–1.088   14.1 -> 15.0 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    enet/prox_in/Float64/n1/normal               1.017–1.088   17.1 -> 18.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_out/Float32/n8/normal               1.028–1.083   10.6 -> 11.2 ns   output offsets [64, 64, 64, 256, 64, 64]
  some    l1/prox_incopy/Float64/n1/normal             1.044–1.079   11.2 -> 12.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n8/normal              1.007–1.075   29.5 -> 30.2 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    l1/prox_alloc/Float32/n8/normal              1.022–1.071   27.8 -> 28.8 ns   output offsets [64, 64, 64, 256, 64, 64]
  some    l1/block_one/Float64/n1/normal               1.000–1.071   4.1 -> 4.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float64/n1/normal              1.017–1.070   17.1 -> 17.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n4/normal            0.998–1.059   34.9 -> 35.7 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/valprox_out/Float32/n64/normal            0.854–1.058   26.8 -> 23.3 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    enet/prox_incopy/Float32/n1/normal           1.028–1.056   20.9 -> 22.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n1/normal            1.006–1.055   33.1 -> 34.2 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float32/n4/normal              1.033–1.055   26.5 -> 27.6 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1w/valprox_out/Float64/n2/normal            0.967–1.051   17.7 -> 17.6 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_alloc/Float32/n2/normal              1.016–1.050   26.1 -> 27.0 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    enetw/prox_out/Float32/n1/normal             1.000–1.050   17.8 -> 18.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_in/Float32/n1/normal              1.000–1.050   17.8 -> 18.6 ns   output offsets [64, 64, 64, 64, 64, 64]

## xeon4116-julia1.13.4 (6 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 4032 of 4032

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3d | A3dr | A3m | control |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 1.00 [0.96–1.07] | 0.99 [0.93–1.07] | 0.98 [0.93–1.07] | 0.99 [0.93–1.07] | 1.00 |
| enet | block_loop | medium | 4 | 0.85 [0.82–0.90] | 0.87 [0.82–0.96] | 0.86 [0.82–0.90] | 0.86 [0.82–0.91] | 1.00 |
| enet | block_one | short | 2 | 0.99 [0.94–1.04] | 0.99 [0.92–1.06] | 1.01 [0.98–1.04] | 0.98 [0.97–1.00] | 0.98 |
| enet | prox_alloc | short | 14 | 1.00 [0.98–1.03] | 1.00 [0.98–1.03] | 1.00 [0.98–1.02] | 0.99 [0.97–1.01] | 1.00 |
| enet | prox_alloc | medium | 14 | 1.01 [0.97–1.07] | 1.00 [0.96–1.03] | 1.01 [0.96–1.05] | 1.01 [0.96–1.03] | 1.00 |
| enet | prox_alloc | large | 12 | 1.03 [1.01–1.05] | 1.00 [0.99–1.01] | 1.03 [1.01–1.06] | 1.03 [0.98–1.06] | 1.00 |
| enet | prox_g_out | short | 14 | 1.01 [0.99–1.06] | 1.00 [0.98–1.03] | 1.01 [1.00–1.08] | 1.01 [1.00–1.05] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_in | short | 14 | 0.91 [0.82–1.03] | 0.93 [0.75–1.08] | 0.91 [0.74–1.07] | 0.93 [0.75–1.08] | 1.00 |
| enet | prox_in | medium | 14 | 0.77 [0.76–0.79] | 0.77 [0.76–0.80] | 0.77 [0.75–0.79] | 0.78 [0.77–0.79] | 1.00 |
| enet | prox_in | large | 12 | 0.78 [0.77–0.80] | 0.78 [0.76–0.80] | 0.78 [0.77–0.80] | 0.79 [0.76–0.81] | 1.01 |
| enet | prox_incopy | short | 14 | 0.92 [0.82–1.02] | 0.94 [0.83–1.05] | 0.93 [0.82–1.02] | 0.95 [0.84–1.07] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.80 [0.77–0.82] | 0.79 [0.77–0.84] | 0.79 [0.77–0.83] | 0.79 [0.77–0.85] | 1.00 |
| enet | prox_incopy | large | 12 | 0.83 [0.78–0.94] | 0.82 [0.78–0.93] | 0.83 [0.78–0.94] | 0.83 [0.78–0.93] | 1.01 |
| enet | prox_out | short | 14 | 0.96 [0.85–1.05] | 1.02 [0.89–1.08] | 1.03 [0.90–1.13] | 1.02 [0.87–1.08] | 0.98 |
| enet | prox_out | medium | 14 | 0.84 [0.82–0.87] | 0.87 [0.80–1.00] | 0.84 [0.79–0.90] | 0.83 [0.79–0.90] | 1.00 |
| enet | prox_out | large | 12 | 0.89 [0.81–1.02] | 1.00 [1.00–1.00] | 0.89 [0.81–1.02] | 0.89 [0.81–1.02] | 1.00 |
| enet | sparse_prox | large | 2 | 0.90 [0.86–0.94] | 0.91 [0.85–0.96] | 0.91 [0.85–0.96] | 0.90 [0.86–0.95] | 1.02 |
| enet | valprox_out | short | 14 | 0.98 [0.91–1.03] | 1.01 [0.92–1.06] | 1.02 [0.92–1.08] | 1.02 [0.93–1.06] | 0.98 |
| enet | valprox_out | medium | 14 | 0.88 [0.87–0.91] | 0.91 [0.87–1.00] | 0.89 [0.87–0.92] | 0.88 [0.87–0.91] | 1.00 |
| enet | valprox_out | large | 12 | 0.92 [0.86–1.02] | 1.00 [1.00–1.00] | 0.92 [0.86–1.02] | 0.92 [0.86–1.02] | 1.00 |
| enetw | block_loop | short | 4 | 1.01 [0.97–1.04] | 1.03 [0.99–1.09] | 1.01 [0.97–1.04] | 1.02 [0.96–1.07] | 1.01 |
| enetw | block_loop | medium | 4 | 0.91 [0.89–0.94] | 0.91 [0.89–0.95] | 0.91 [0.89–0.94] | 0.92 [0.89–0.96] | 1.01 |
| enetw | block_one | short | 2 | 1.05 [1.01–1.10] | 1.03 [1.01–1.06] | 1.02 [1.02–1.03] | 1.03 [1.00–1.06] | 1.02 |
| enetw | prox_alloc | short | 14 | 0.99 [0.98–1.01] | 0.99 [0.98–1.00] | 0.99 [0.98–1.00] | 1.00 [0.98–1.02] | 1.00 |
| enetw | prox_alloc | medium | 14 | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 1.00 [0.98–1.01] | 1.00 [0.99–1.02] | 1.00 |
| enetw | prox_alloc | large | 12 | 1.01 [1.00–1.02] | 1.00 [0.99–1.01] | 1.01 [1.00–1.02] | 1.01 [0.99–1.03] | 1.00 |
| enetw | prox_g_out | short | 14 | 1.00 [0.98–1.03] | 1.00 [0.96–1.02] | 1.00 [0.98–1.01] | 1.00 [0.98–1.01] | 1.01 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_in | short | 14 | 0.95 [0.87–1.01] | 0.94 [0.87–1.01] | 0.93 [0.86–1.00] | 0.94 [0.87–1.01] | 1.00 |
| enetw | prox_in | medium | 14 | 0.84 [0.84–0.86] | 0.84 [0.83–0.86] | 0.84 [0.84–0.86] | 0.84 [0.84–0.86] | 1.00 |
| enetw | prox_in | large | 12 | 0.85 [0.83–0.86] | 0.85 [0.83–0.87] | 0.85 [0.83–0.86] | 0.84 [0.84–0.86] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.95 [0.88–1.02] | 0.95 [0.88–1.01] | 0.95 [0.88–1.02] | 0.96 [0.88–1.03] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.86 [0.84–0.88] | 0.86 [0.84–0.89] | 0.86 [0.84–0.89] | 0.86 [0.84–0.89] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.88 [0.85–0.95] | 0.87 [0.85–0.95] | 0.87 [0.85–0.95] | 0.88 [0.85–0.95] | 1.00 |
| enetw | prox_out | short | 14 | 0.98 [0.93–1.01] | 0.96 [0.91–1.00] | 0.96 [0.92–1.04] | 0.97 [0.92–1.01] | 0.99 |
| enetw | prox_out | medium | 14 | 0.92 [0.90–0.95] | 0.93 [0.89–1.00] | 0.92 [0.89–0.94] | 0.92 [0.89–0.94] | 1.00 |
| enetw | prox_out | large | 12 | 0.96 [0.91–1.00] | 1.00 [1.00–1.00] | 0.96 [0.92–1.00] | 0.96 [0.92–1.00] | 1.00 |
| enetw | sparse_prox | large | 2 | 0.98 [0.98–0.98] | 0.98 [0.98–0.98] | 0.98 [0.98–0.98] | 0.98 [0.98–0.99] | 1.00 |
| enetw | valprox_out | short | 14 | 1.00 [0.97–1.01] | 0.99 [0.93–1.04] | 1.00 [0.96–1.05] | 0.99 [0.95–1.01] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.95 [0.93–0.97] | 0.96 [0.93–1.00] | 0.95 [0.93–0.99] | 0.95 [0.93–0.98] | 1.00 |
| enetw | valprox_out | large | 12 | 0.97 [0.94–1.00] | 1.00 [1.00–1.00] | 0.97 [0.94–1.00] | 0.97 [0.94–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 0.94 [0.90–0.97] | 0.98 [0.91–1.02] | 1.01 [0.96–1.06] | 0.98 [0.93–1.01] | 0.96 |
| l1 | block_loop | medium | 4 | 0.79 [0.77–0.81] | 0.78 [0.77–0.79] | 0.78 [0.77–0.79] | 0.78 [0.77–0.79] | 1.00 |
| l1 | block_one | short | 2 | 0.92 [0.92–0.92] | 0.92 [0.92–0.92] | 0.92 [0.92–0.92] | 0.92 [0.92–0.92] | 1.00 |
| l1 | prox_alloc | short | 14 | 1.00 [0.98–1.02] | 1.02 [0.98–1.04] | 1.03 [0.99–1.06] | 1.02 [0.98–1.05] | 1.00 |
| l1 | prox_alloc | medium | 14 | 1.00 [0.98–1.02] | 1.00 [0.99–1.02] | 1.01 [0.99–1.03] | 1.01 [0.98–1.03] | 1.00 |
| l1 | prox_alloc | large | 12 | 1.03 [1.00–1.06] | 1.00 [0.98–1.03] | 1.03 [1.01–1.07] | 1.02 [1.00–1.04] | 1.00 |
| l1 | prox_g_out | short | 14 | 0.96 [0.86–1.07] | 0.97 [0.92–1.04] | 0.93 [0.85–1.06] | 0.96 [0.92–1.04] | 1.01 |
| l1 | prox_g_out | medium | 14 | 0.92 [0.84–1.00] | 0.96 [0.93–1.00] | 0.89 [0.84–0.94] | 0.96 [0.93–1.00] | 1.01 |
| l1 | prox_g_out | large | 12 | 0.92 [0.84–1.00] | 0.97 [0.93–1.00] | 0.89 [0.84–0.94] | 0.97 [0.93–1.00] | 1.01 |
| l1 | prox_in | short | 14 | 0.84 [0.73–1.05] | 0.84 [0.73–1.05] | 0.84 [0.73–1.05] | 0.84 [0.73–1.05] | 0.97 |
| l1 | prox_in | medium | 14 | 0.75 [0.71–0.80] | 0.75 [0.71–0.79] | 0.75 [0.71–0.80] | 0.74 [0.71–0.80] | 0.97 |
| l1 | prox_in | large | 12 | 0.74 [0.72–0.77] | 0.75 [0.72–0.77] | 0.75 [0.72–0.77] | 0.74 [0.72–0.76] | 0.97 |
| l1 | prox_incopy | short | 14 | 0.88 [0.73–1.03] | 0.90 [0.73–1.09] | 0.90 [0.73–1.09] | 0.91 [0.74–1.13] | 0.99 |
| l1 | prox_incopy | medium | 14 | 0.78 [0.75–0.80] | 0.78 [0.75–0.81] | 0.78 [0.74–0.81] | 0.78 [0.74–0.81] | 0.98 |
| l1 | prox_incopy | large | 12 | 0.81 [0.77–0.91] | 0.81 [0.76–0.91] | 0.81 [0.77–0.91] | 0.81 [0.77–0.91] | 0.98 |
| l1 | prox_out | short | 14 | 0.96 [0.84–1.08] | 0.99 [0.83–1.12] | 1.04 [0.85–1.20] | 1.02 [0.84–1.15] | 1.01 |
| l1 | prox_out | medium | 14 | 0.78 [0.76–0.81] | 0.83 [0.76–1.00] | 0.79 [0.76–0.86] | 0.78 [0.76–0.83] | 1.00 |
| l1 | prox_out | large | 12 | 0.86 [0.76–1.01] | 1.00 [1.00–1.00] | 0.86 [0.77–1.01] | 0.86 [0.77–1.00] | 1.00 |
| l1 | sparse_prox | large | 2 | 0.98 [0.96–0.99] | 0.97 [0.96–0.99] | 0.97 [0.96–0.99] | 0.97 [0.96–0.99] | 1.00 |
| l1 | valprox_out | short | 14 | 0.98 [0.88–1.04] | 0.99 [0.89–1.05] | 1.03 [0.91–1.11] | 1.00 [0.89–1.05] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.85 [0.83–0.89] | 0.88 [0.82–1.00] | 0.85 [0.82–0.91] | 0.85 [0.82–0.90] | 1.00 |
| l1 | valprox_out | large | 12 | 0.90 [0.83–1.00] | 1.00 [1.00–1.00] | 0.90 [0.83–1.00] | 0.90 [0.83–1.00] | 1.00 |
| l1w | block_loop | short | 4 | 0.99 [0.97–1.02] | 0.98 [0.96–1.01] | 0.99 [0.97–1.02] | 0.98 [0.96–1.03] | 0.99 |
| l1w | block_loop | medium | 4 | 0.87 [0.85–0.88] | 0.86 [0.85–0.88] | 0.87 [0.85–0.88] | 0.87 [0.85–0.88] | 1.00 |
| l1w | block_one | short | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 1.00 [0.98–1.02] | 1.00 [0.97–1.03] | 1.00 [0.99–1.02] | 1.00 [0.98–1.02] | 1.00 |
| l1w | prox_alloc | medium | 14 | 1.00 [0.97–1.01] | 1.00 [0.98–1.01] | 1.00 [0.99–1.00] | 1.00 [0.98–1.01] | 1.00 |
| l1w | prox_alloc | large | 12 | 1.01 [1.00–1.02] | 1.00 [1.00–1.01] | 1.01 [1.00–1.03] | 1.01 [1.00–1.02] | 1.00 |
| l1w | prox_g_out | short | 14 | 0.90 [0.79–1.03] | 0.89 [0.79–1.02] | 0.89 [0.79–1.02] | 0.89 [0.79–1.02] | 0.95 |
| l1w | prox_g_out | medium | 14 | 0.84 [0.82–0.86] | 0.84 [0.83–0.85] | 0.84 [0.82–0.86] | 0.84 [0.83–0.85] | 0.98 |
| l1w | prox_g_out | large | 12 | 0.86 [0.82–0.97] | 0.86 [0.83–0.96] | 0.86 [0.83–0.97] | 0.86 [0.83–0.96] | 0.99 |
| l1w | prox_in | short | 14 | 0.94 [0.87–1.09] | 0.94 [0.87–1.04] | 0.95 [0.88–1.13] | 0.94 [0.87–1.14] | 1.02 |
| l1w | prox_in | medium | 14 | 0.85 [0.83–0.86] | 0.85 [0.84–0.86] | 0.85 [0.84–0.86] | 0.83 [0.83–0.85] | 1.00 |
| l1w | prox_in | large | 12 | 0.85 [0.83–0.88] | 0.86 [0.85–0.89] | 0.85 [0.83–0.89] | 0.85 [0.83–0.89] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.93 [0.84–1.00] | 0.96 [0.87–1.10] | 0.95 [0.86–1.09] | 0.95 [0.85–1.14] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.85 [0.84–0.86] | 0.86 [0.85–0.87] | 0.86 [0.85–0.87] | 0.84 [0.83–0.86] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.88 [0.85–0.96] | 0.88 [0.86–0.96] | 0.88 [0.85–0.95] | 0.87 [0.84–0.95] | 1.00 |
| l1w | prox_out | short | 14 | 0.96 [0.88–1.07] | 0.99 [0.90–1.06] | 1.02 [0.90–1.17] | 1.00 [0.90–1.18] | 1.01 |
| l1w | prox_out | medium | 14 | 0.88 [0.85–0.92] | 0.91 [0.85–1.00] | 0.88 [0.85–0.92] | 0.89 [0.85–0.92] | 1.00 |
| l1w | prox_out | large | 12 | 0.96 [0.90–1.01] | 1.00 [1.00–1.00] | 0.96 [0.90–1.01] | 0.96 [0.90–1.01] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.01 [1.00–1.01] | 1.00 [1.00–1.00] | 1.00 |
| l1w | valprox_out | short | 14 | 0.98 [0.93–1.02] | 1.00 [0.93–1.04] | 1.02 [0.94–1.08] | 1.00 [0.94–1.04] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.93 [0.91–0.95] | 0.94 [0.91–1.00] | 0.93 [0.91–0.95] | 0.93 [0.91–0.95] | 1.00 |
| l1w | valprox_out | large | 12 | 0.97 [0.94–1.01] | 1.00 [1.00–1.00] | 0.97 [0.94–1.01] | 0.97 [0.94–1.01] | 1.00 |

A3M4: 0 of 1008 cells slower in every process, 235 faster in every process, 6 above 1.05 in some processes only, 208 undecided (control unclean)
  some    enet/prox_g_out/Float64/n8/normal            0.994–1.162   31.0 -> 31.1 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    enetw/valprox_out/Float64/n64/normal         0.950–1.118   61.9 -> 61.9 ns   output offsets [544, 544, 544, 544, 544, 544]
  some    enet/prox_alloc/Float32/n100000/normal       1.019–1.069   34826.6 -> 37208.6 ns   output offsets [2816, 2752, 2752, 2752, 2752, 2752]
  some    enet/prox_alloc/Float32/n32768/normal        1.033–1.064   11635.0 -> 12202.5 ns   output offsets [2880, 896, 1408, 576, 3712, 1472]
  some    l1/prox_alloc/Float64/n65536/normal          1.020–1.053   51589.3 -> 52852.9 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float32/n2/normal              0.936–1.050   16.7 -> 16.4 ns   output offsets [64, 64, 64, 64, 64, 64]

A3d: 0 of 1008 cells slower in every process, 175 faster in every process, 11 above 1.05 in some processes only, 201 undecided (control unclean)
  some    l1w/block_one/Float64/n1/normal              1.000–1.162   4.4 -> 4.4 ns   output offsets [64, 64, 64, 64, 64, 96]
  some    l1/valprox_out/Float64/n128/normal           0.873–1.142   59.6 -> 53.2 ns   output offsets [1088, 1088, 1088, 1088, 1088, 1088]
  some    l1/prox_in/Float64/n1/normal                 1.048–1.095   7.0 -> 7.4 ns   output offsets [64, 64, 64, 64, 64, 96]
  some    enet/prox_out/Float32/n2/normal              0.969–1.080   16.7 -> 17.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/valprox_out/Float64/n4/normal            1.001–1.079   19.3 -> 19.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float32/n4/normal             1.008–1.076   30.7 -> 31.4 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_alloc/Float32/n1/normal              1.018–1.064   27.1 -> 28.3 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n1/normal              1.030–1.064   26.5 -> 27.7 ns   output offsets [64, 64, 64, 64, 64, 96]
  some    l1w/valprox_out/Float32/n1/normal            1.009–1.064   14.9 -> 15.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_in/Float32/n4/normal               0.911–1.059   19.4 -> 20.0 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enetw/prox_alloc/Float64/n8192/normal        0.987–1.051   6515.3 -> 6438.2 ns   output offsets [3328, 512, 960, 64, 1792, 960]

A3dr: 3 of 1008 cells slower in every process, 234 faster in every process, 14 above 1.05 in some processes only, 209 undecided (control unclean)
  slower  l1w/valprox_out/Float64/n1/normal            1.054–1.079   14.9 -> 16.0 ns
  slower  l1/prox_alloc/Float64/n1/normal              1.053–1.076   26.5 -> 28.0 ns
  slower  l1w/valprox_out/Float32/n1/normal            1.053–1.067   14.9 -> 15.8 ns
  some    l1/prox_in/Float64/n1/normal                 1.048–1.116   7.0 -> 7.4 ns   output offsets [64, 64, 64, 64, 64, 96]
  some    l1w/valprox_out/Float64/n4/normal            1.016–1.094   19.3 -> 20.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float32/n2/normal              0.988–1.085   16.7 -> 17.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float32/n4/normal             0.986–1.074   30.7 -> 30.8 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_alloc/Float32/n100000/normal       1.007–1.073   34826.6 -> 36479.0 ns   output offsets [2816, 2752, 2752, 2752, 2752, 2752]
  some    l1w/valprox_out/Float32/n32/normal           0.975–1.072   24.3 -> 25.4 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    l1/prox_alloc/Float32/n1/normal              1.038–1.071   27.1 -> 28.9 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n32768/normal        1.024–1.064   11635.0 -> 12175.8 ns   output offsets [2880, 896, 1408, 576, 3712, 1472]
  some    enet/prox_in/Float32/n4/normal               0.910–1.064   19.4 -> 18.9 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_g_out/Float64/n8/normal            0.995–1.063   31.0 -> 31.6 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    enet/prox_alloc/Float64/n4096/normal         1.018–1.054   3413.0 -> 3499.6 ns   output offsets [1920, 256, 1728, 704, 3712, 3904]
  some    l1/prox_alloc/Float64/n65536/normal          1.030–1.053   51589.3 -> 53619.4 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float32/n2/normal              1.016–1.052   28.1 -> 29.3 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n4096/normal           0.991–1.052   3409.6 -> 3537.8 ns   output offsets [1920, 256, 1728, 704, 3712, 3904]

A3m: 0 of 1008 cells slower in every process, 239 faster in every process, 13 above 1.05 in some processes only, 212 undecided (control unclean)
  some    l1w/prox_alloc/Float32/n4/normal             0.982–1.104   30.7 -> 30.7 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_in/Float64/n1/normal                 1.048–1.097   7.0 -> 7.4 ns   output offsets [64, 64, 64, 64, 64, 96]
  some    enet/prox_out/Float32/n2/normal              1.019–1.094   16.7 -> 18.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/valprox_out/Float64/n1/normal            1.011–1.089   14.9 -> 15.7 ns   output offsets [64, 64, 64, 64, 64, 96]
  some    enet/prox_alloc/Float32/n100000/normal       1.035–1.072   34826.6 -> 36944.7 ns   output offsets [2816, 2752, 2752, 2752, 2752, 2752]
  some    l1w/valprox_out/Float64/n4/normal            0.997–1.070   19.3 -> 19.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_alloc/Float32/n16/normal          0.985–1.068   47.1 -> 47.9 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    l1/block_one/Float32/n1/normal               0.917–1.066   4.0 -> 3.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/valprox_out/Float64/n8/normal            0.997–1.064   21.3 -> 21.3 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    l1/prox_alloc/Float64/n65536/normal          0.992–1.063   51589.3 -> 52423.6 ns   output offsets [128, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n1/normal              1.039–1.062   26.5 -> 27.7 ns   output offsets [64, 64, 64, 64, 64, 96]
  some    enet/prox_alloc/Float32/n32768/normal        1.030–1.058   11635.0 -> 12277.0 ns   output offsets [2880, 896, 1408, 576, 3712, 1472]
  some    l1w/prox_alloc/Float64/n2/normal             0.987–1.056   29.2 -> 30.1 ns   output offsets [48, 48, 48, 48, 48, 48]
