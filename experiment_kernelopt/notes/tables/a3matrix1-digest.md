
## epyc7513a-julia1.13.4 (3 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 5000 of 5000

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3c | A3cr | A3i | Mi | control |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 0.99 [0.95–1.03] | 0.99 [0.94–1.03] | 0.99 [0.94–1.03] | 0.99 [0.94–1.03] | 1.01 [1.00–1.03] | 1.00 |
| enet | block_loop | medium | 4 | 0.73 [0.69–0.81] | 0.73 [0.69–0.80] | 0.74 [0.70–0.80] | 0.74 [0.69–0.80] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_alloc | short | 14 | 0.98 [0.84–1.00] | 0.98 [0.84–1.00] | 0.98 [0.85–1.01] | 0.99 [0.86–1.02] | 1.02 [1.00–1.15] | 1.00 |
| enet | prox_alloc | medium | 14 | 0.91 [0.70–0.98] | 0.92 [0.83–1.01] | 0.97 [0.91–1.16] | 0.93 [0.87–0.98] | 1.02 [0.96–1.28] | 1.02 |
| enet | prox_alloc | large | 12 | 0.94 [0.75–1.09] | 0.96 [0.75–1.12] | 0.96 [0.78–1.17] | 0.99 [0.92–1.19] | 1.00 [0.88–1.08] | 1.05 |
| enet | prox_g_out | short | 14 | 1.00 [0.98–1.02] | 1.00 [0.97–1.03] | 1.00 [0.96–1.03] | 1.11 [1.00–1.27] | 1.11 [1.00–1.28] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [0.99–1.00] | 1.00 [0.99–1.00] | 1.00 [0.99–1.00] | 1.31 [1.24–1.38] | 1.31 [1.24–1.38] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [0.99–1.00] | 1.00 [0.99–1.00] | 1.32 [1.25–1.38] | 1.32 [1.25–1.38] | 1.00 |
| enet | prox_in | short | 14 | 0.86 [0.71–1.00] | 0.87 [0.71–1.00] | 0.88 [0.72–1.03] | 0.63 [0.22–1.03] | 0.67 [0.24–1.06] | 1.00 |
| enet | prox_in | medium | 14 | 0.70 [0.69–0.71] | 0.73 [0.69–1.00] | 0.74 [0.69–1.00] | 0.13 [0.09–0.22] | 0.19 [0.13–0.29] | 1.00 |
| enet | prox_in | large | 12 | 0.69 [0.69–0.69] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 0.12 [0.08–0.17] | 0.18 [0.12–0.25] | 1.00 |
| enet | prox_incopy | short | 14 | 0.88 [0.73–1.00] | 0.89 [0.74–1.00] | 0.90 [0.74–1.01] | 0.69 [0.28–1.02] | 0.72 [0.30–1.06] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.72 [0.71–0.74] | 0.75 [0.71–1.00] | 0.76 [0.71–1.00] | 0.19 [0.12–0.30] | 0.25 [0.16–0.36] | 1.00 |
| enet | prox_incopy | large | 12 | 0.73 [0.72–0.75] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 0.22 [0.16–0.31] | 0.27 [0.19–0.38] | 1.00 |
| enet | prox_out | short | 14 | 0.95 [0.73–1.00] | 0.95 [0.73–1.01] | 0.97 [0.74–1.03] | 0.98 [0.75–1.06] | 1.03 [1.00–1.06] | 1.00 |
| enet | prox_out | medium | 14 | 0.71 [0.68–0.81] | 0.74 [0.68–1.00] | 0.76 [0.69–1.00] | 0.71 [0.68–0.86] | 1.00 [1.00–1.02] | 1.00 |
| enet | prox_out | large | 12 | 0.91 [0.71–1.34] | 1.00 [0.99–1.01] | 1.00 [0.99–1.00] | 0.90 [0.71–1.32] | 1.00 [0.99–1.01] | 1.00 |
| enet | sparse_prox | large | 2 | 1.10 [1.09–1.10] | 1.10 [1.10–1.10] | 1.10 [1.09–1.10] | 1.05 [1.01–1.10] | 1.01 [1.00–1.01] | 1.00 |
| enet | valprox_out | short | 14 | 0.98 [0.86–1.02] | 0.98 [0.87–1.03] | 1.00 [0.87–1.09] | 0.99 [0.87–1.02] | 1.01 [0.99–1.04] | 1.00 |
| enet | valprox_out | medium | 14 | 0.79 [0.76–0.88] | 0.82 [0.76–1.00] | 0.82 [0.76–1.00] | 0.79 [0.76–0.90] | 1.00 [1.00–1.02] | 1.00 |
| enet | valprox_out | large | 12 | 0.93 [0.78–1.25] | 1.00 [0.99–1.01] | 1.00 [0.99–1.00] | 0.93 [0.78–1.23] | 1.00 [0.99–1.00] | 1.00 |
| enetw | block_loop | short | 4 | 1.01 [0.90–1.11] | 1.00 [0.88–1.11] | 1.00 [0.90–1.14] | 0.99 [0.89–1.14] | 1.00 [1.00–1.02] | 1.00 |
| enetw | block_loop | medium | 4 | 0.95 [0.86–1.02] | 0.88 [0.86–0.90] | 0.88 [0.85–0.94] | 0.89 [0.85–0.95] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.98 [0.93–1.01] | 0.99 [0.93–1.01] | 0.99 [0.94–1.01] | 1.01 [0.96–1.10] | 1.02 [1.00–1.17] | 1.00 |
| enetw | prox_alloc | medium | 14 | 0.91 [0.73–1.02] | 0.97 [0.90–1.05] | 0.98 [0.90–1.14] | 0.99 [0.89–1.17] | 1.01 [0.98–1.03] | 1.01 |
| enetw | prox_alloc | large | 12 | 1.02 [0.80–1.31] | 0.98 [0.75–1.24] | 1.00 [0.95–1.06] | 1.05 [1.00–1.30] | 0.98 [0.84–1.01] | 1.01 |
| enetw | prox_g_out | short | 14 | 0.99 [0.97–1.01] | 1.00 [0.98–1.02] | 0.99 [0.98–1.01] | 1.05 [0.98–1.13] | 1.05 [0.99–1.13] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 1.00 [0.99–1.00] | 1.15 [1.14–1.15] | 1.15 [1.14–1.15] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 1.00 [0.99–1.00] | 1.15 [1.14–1.16] | 1.15 [1.14–1.15] | 1.00 |
| enetw | prox_in | short | 14 | 0.91 [0.82–1.02] | 0.92 [0.81–1.02] | 0.92 [0.81–1.02] | 0.64 [0.25–1.05] | 0.66 [0.27–1.05] | 1.00 |
| enetw | prox_in | medium | 14 | 0.78 [0.78–0.80] | 0.81 [0.77–1.00] | 0.81 [0.78–1.00] | 0.15 [0.09–0.26] | 0.20 [0.13–0.31] | 1.00 |
| enetw | prox_in | large | 12 | 0.78 [0.78–0.83] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 0.15 [0.10–0.26] | 0.19 [0.13–0.29] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.93 [0.83–1.04] | 0.93 [0.83–1.02] | 0.93 [0.84–1.04] | 0.69 [0.30–1.04] | 0.70 [0.32–1.04] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.80 [0.79–0.82] | 0.82 [0.79–1.00] | 0.83 [0.79–1.00] | 0.21 [0.13–0.32] | 0.26 [0.17–0.37] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.81 [0.80–0.85] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 0.25 [0.16–0.39] | 0.28 [0.19–0.40] | 1.00 |
| enetw | prox_out | short | 14 | 0.97 [0.87–1.02] | 0.98 [0.87–1.05] | 0.99 [0.87–1.02] | 0.99 [0.88–1.05] | 1.01 [1.00–1.05] | 1.00 |
| enetw | prox_out | medium | 14 | 0.84 [0.74–0.97] | 0.84 [0.73–1.00] | 0.85 [0.73–1.00] | 0.84 [0.74–0.97] | 1.00 [0.99–1.03] | 1.00 |
| enetw | prox_out | large | 12 | 1.03 [0.96–1.21] | 1.00 [1.00–1.00] | 1.00 [0.99–1.00] | 1.03 [0.96–1.21] | 1.00 [0.99–1.00] | 1.00 |
| enetw | sparse_prox | large | 2 | 1.02 [0.98–1.06] | 0.99 [0.94–1.05] | 0.95 [0.93–0.97] | 1.01 [0.95–1.08] | 1.00 [0.99–1.00] | 1.00 |
| enetw | valprox_out | short | 14 | 0.97 [0.85–1.01] | 0.97 [0.87–1.03] | 0.97 [0.86–1.02] | 0.98 [0.84–1.02] | 1.02 [1.00–1.04] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.90 [0.84–0.99] | 0.90 [0.84–1.01] | 0.90 [0.84–1.00] | 0.90 [0.84–0.98] | 1.00 [1.00–1.02] | 1.00 |
| enetw | valprox_out | large | 12 | 1.02 [0.97–1.16] | 1.00 [1.00–1.00] | 1.00 [0.98–1.00] | 1.02 [0.97–1.13] | 1.00 [1.00–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 0.98 [0.96–1.00] | 0.98 [0.96–1.00] | 0.98 [0.96–1.00] | 0.98 [0.96–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | block_loop | medium | 4 | 0.69 [0.67–0.77] | 0.69 [0.67–0.75] | 0.69 [0.67–0.77] | 0.69 [0.67–0.77] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.99 [0.95–1.01] | 1.01 [0.95–1.04] | 1.02 [0.98–1.04] | 1.00 [0.97–1.11] | 1.01 [1.00–1.03] | 1.01 |
| l1 | prox_alloc | medium | 14 | 0.92 [0.68–1.02] | 0.95 [0.91–0.98] | 0.96 [0.90–1.02] | 0.95 [0.91–1.02] | 0.97 [0.80–1.02] | 1.00 |
| l1 | prox_alloc | large | 12 | 0.99 [0.79–1.07] | 0.98 [0.83–1.03] | 0.98 [0.77–1.02] | 1.03 [0.96–1.06] | 1.00 [0.81–1.33] | 1.00 |
| l1 | prox_g_out | short | 14 | 1.01 [0.93–1.05] | 1.01 [0.93–1.05] | 1.00 [0.93–1.05] | 1.13 [1.04–1.33] | 1.05 [1.01–1.20] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.99 [0.97–1.00] | 0.98 [0.98–1.00] | 0.99 [0.97–1.00] | 1.53 [1.43–1.74] | 1.20 [1.12–1.35] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.98 [0.97–0.98] | 0.98 [0.97–0.99] | 0.98 [0.97–0.98] | 1.53 [1.46–1.55] | 1.32 [1.21–1.45] | 1.00 |
| l1 | prox_in | short | 14 | 0.83 [0.68–1.00] | 0.85 [0.68–1.06] | 0.85 [0.68–1.06] | 0.53 [0.14–1.00] | 0.57 [0.17–1.06] | 1.00 |
| l1 | prox_in | medium | 14 | 0.68 [0.68–0.69] | 0.72 [0.68–1.00] | 0.72 [0.68–1.00] | 0.12 [0.08–0.17] | 0.18 [0.12–0.25] | 1.00 |
| l1 | prox_in | large | 12 | 0.68 [0.68–0.68] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 0.12 [0.08–0.17] | 0.18 [0.12–0.25] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.88 [0.70–1.03] | 0.90 [0.72–1.12] | 0.92 [0.72–1.16] | 0.62 [0.22–1.00] | 0.64 [0.24–1.06] | 1.00 |
| l1 | prox_incopy | medium | 14 | 0.71 [0.70–0.73] | 0.74 [0.70–1.00] | 0.74 [0.70–1.00] | 0.18 [0.12–0.29] | 0.23 [0.16–0.36] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.73 [0.71–0.74] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 0.22 [0.15–0.31] | 0.27 [0.19–0.38] | 1.00 |
| l1 | prox_out | short | 14 | 0.96 [0.77–1.05] | 0.98 [0.79–1.05] | 1.00 [0.79–1.10] | 0.97 [0.79–1.06] | 1.02 [1.00–1.05] | 1.00 |
| l1 | prox_out | medium | 14 | 0.68 [0.67–0.79] | 0.72 [0.67–1.00] | 0.72 [0.67–1.00] | 0.68 [0.67–0.79] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_out | large | 12 | 0.81 [0.67–0.97] | 0.98 [0.90–1.00] | 1.00 [1.00–1.01] | 0.81 [0.67–0.97] | 1.00 [1.00–1.01] | 1.00 |
| l1 | sparse_prox | large | 2 | 1.02 [1.02–1.02] | 1.00 [0.97–1.03] | 1.02 [1.02–1.03] | 1.03 [1.02–1.03] | 1.01 [1.01–1.01] | 1.01 |
| l1 | valprox_out | short | 14 | 0.96 [0.86–1.01] | 0.96 [0.88–1.00] | 0.99 [0.88–1.08] | 0.96 [0.86–1.00] | 1.00 [0.97–1.02] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.77 [0.74–0.86] | 0.80 [0.74–1.00] | 0.80 [0.74–1.00] | 0.77 [0.74–0.86] | 1.00 [1.00–1.02] | 1.00 |
| l1 | valprox_out | large | 12 | 0.86 [0.76–0.99] | 0.99 [0.94–1.00] | 1.00 [0.99–1.00] | 0.86 [0.76–0.99] | 1.00 [1.00–1.01] | 1.00 |
| l1w | block_loop | short | 4 | 0.98 [0.90–1.09] | 0.96 [0.90–1.05] | 0.97 [0.93–1.00] | 0.98 [0.93–1.05] | 1.00 [1.00–1.00] | 0.99 |
| l1w | block_loop | medium | 4 | 0.72 [0.68–0.78] | 0.72 [0.69–0.78] | 0.72 [0.68–0.76] | 0.72 [0.68–0.78] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 0.99 [0.93–1.02] | 0.99 [0.92–1.03] | 0.99 [0.93–1.02] | 1.00 [0.94–1.03] | 1.01 [1.00–1.03] | 1.01 |
| l1w | prox_alloc | medium | 14 | 0.94 [0.73–1.10] | 0.96 [0.89–1.10] | 0.98 [0.89–1.10] | 0.99 [0.89–1.10] | 0.98 [0.84–1.04] | 1.01 |
| l1w | prox_alloc | large | 12 | 1.07 [0.87–1.29] | 0.97 [0.76–1.11] | 0.96 [0.82–1.01] | 1.07 [0.94–1.29] | 1.00 [0.78–1.21] | 1.03 |
| l1w | prox_g_out | short | 14 | 0.99 [0.91–1.04] | 1.05 [1.00–1.10] | 1.05 [1.00–1.10] | 1.01 [0.91–1.08] | 1.03 [0.99–1.10] | 1.03 |
| l1w | prox_g_out | medium | 14 | 0.96 [0.96–0.98] | 1.10 [1.04–1.11] | 1.10 [1.09–1.11] | 0.96 [0.96–0.97] | 1.05 [1.00–1.11] | 1.04 |
| l1w | prox_g_out | large | 12 | 0.96 [0.96–0.97] | 1.09 [1.03–1.10] | 1.09 [1.03–1.10] | 0.96 [0.96–0.97] | 1.05 [1.00–1.10] | 1.04 |
| l1w | prox_in | short | 14 | 0.89 [0.75–1.00] | 0.90 [0.76–1.00] | 0.92 [0.76–1.05] | 0.54 [0.16–1.05] | 0.57 [0.19–1.10] | 1.00 |
| l1w | prox_in | medium | 14 | 0.72 [0.71–0.73] | 0.75 [0.71–1.00] | 0.75 [0.71–1.00] | 0.13 [0.09–0.19] | 0.19 [0.13–0.26] | 1.00 |
| l1w | prox_in | large | 12 | 0.71 [0.71–0.71] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 0.15 [0.09–0.30] | 0.18 [0.13–0.27] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.90 [0.77–1.00] | 0.94 [0.78–1.06] | 0.94 [0.79–1.06] | 0.61 [0.23–1.03] | 0.64 [0.27–1.01] | 0.99 |
| l1w | prox_incopy | medium | 14 | 0.74 [0.72–0.76] | 0.77 [0.72–1.00] | 0.77 [0.72–1.00] | 0.20 [0.13–0.31] | 0.25 [0.17–0.36] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.75 [0.73–0.76] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 0.25 [0.16–0.42] | 0.28 [0.19–0.39] | 1.00 |
| l1w | prox_out | short | 14 | 0.98 [0.83–1.04] | 0.97 [0.82–1.04] | 0.96 [0.81–1.00] | 0.99 [0.81–1.09] | 1.03 [1.00–1.05] | 1.02 |
| l1w | prox_out | medium | 14 | 0.79 [0.67–0.99] | 0.79 [0.67–1.00] | 0.79 [0.67–1.00] | 0.79 [0.67–0.99] | 1.00 [0.99–1.00] | 1.00 |
| l1w | prox_out | large | 12 | 1.01 [0.99–1.07] | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 1.01 [0.99–1.06] | 1.00 [1.00–1.01] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.01 [1.00–1.02] | 1.02 [1.02–1.02] | 1.01 [1.00–1.03] | 1.02 [1.01–1.03] | 1.00 [0.99–1.01] | 1.00 |
| l1w | valprox_out | short | 14 | 0.96 [0.90–1.00] | 0.98 [0.90–1.03] | 0.99 [0.90–1.03] | 0.99 [0.89–1.03] | 1.02 [1.00–1.03] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.89 [0.83–0.99] | 0.89 [0.83–1.00] | 0.89 [0.83–1.00] | 0.89 [0.82–0.99] | 1.00 [1.00–1.01] | 1.00 |
| l1w | valprox_out | large | 12 | 1.01 [0.99–1.06] | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 1.01 [0.99–1.05] | 1.00 [0.99–1.00] | 1.00 |

A3M4: 4 of 1000 cells slower in every process, 398 faster in every process, 29 above 1.05 in some processes only, 64 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 10.9 ns
  slower  enet/prox_out/Float32/n1048576/normal        1.070–1.363   110458.7 -> 118669.5 ns
  slower  enetw/prox_out/Float32/n1048576/normal       1.066–1.147   138186.1 -> 146604.7 ns
  slower  l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns
  some    enet/prox_out/Float32/n262144/normal         0.853–1.362   27812.0 -> 35533.6 ns   output offsets [64, 64, 64]
  some    enet/prox_out/Float64/n32768/normal          0.800–1.284   6997.3 -> 5770.8 ns   output offsets [64, 64, 128]
  some    enetw/prox_out/Float32/n65536/normal         0.991–1.268   8081.2 -> 9778.3 ns   output offsets [2944, 64, 128]
  some    enet/valprox_out/Float32/n262144/normal      0.901–1.261   39074.6 -> 46703.2 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float32/n1048576/normal     1.045–1.257   158977.1 -> 168668.6 ns   output offsets [64, 64, 64]
  some    enet/prox_out/Float64/n65536/normal          0.835–1.254   14826.3 -> 18644.9 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float32/n65536/normal       0.802–1.249   9527.1 -> 11880.7 ns   output offsets [2944, 64, 128]
  some    enetw/valprox_out/Float32/n65536/normal      0.994–1.199   12756.3 -> 14512.9 ns   output offsets [2944, 64, 128]
  some    l1w/prox_out/Float32/n65536/normal           1.002–1.183   8432.9 -> 9117.3 ns   output offsets [2944, 64, 128]
  some    enet/valprox_out/Float64/n65536/normal       0.890–1.154   21017.1 -> 23886.4 ns   output offsets [64, 64, 64]
  some    enetw/prox_out/Float32/n262144/normal        0.999–1.149   34677.7 -> 38466.5 ns   output offsets [64, 64, 64]
  some    enet/prox_alloc/Float32/n16/normal           0.979–1.145   25.7 -> 25.5 ns   output offsets [96, 96, 96]
  some    l1/valprox_out/Float32/n65536/normal         0.791–1.138   9666.6 -> 9854.9 ns   output offsets [2944, 64, 128]
  some    enet/prox_alloc/Float32/n8/normal            0.988–1.130   25.0 -> 24.8 ns   output offsets [64, 64, 64]
  some    enet/prox_g_out/Float64/n8/normal            0.997–1.128   20.1 -> 20.4 ns   output offsets [96, 96, 96]
  some    l1w/prox_alloc/Float64/n16384/normal         0.939–1.121   5548.1 -> 6185.8 ns   output offsets [1728, 0, 2688]
  some    l1w/valprox_out/Float32/n65536/normal        1.008–1.114   14014.8 -> 14835.8 ns   output offsets [2944, 64, 128]
  some    enetw/valprox_out/Float32/n262144/normal     0.998–1.094   57341.9 -> 62308.5 ns   output offsets [64, 64, 64]
  some    l1w/block_loop/Float32/n4/normal             1.046–1.091   6.0 -> 6.6 ns   output offsets [48, 48, 48]
  some    enetw/valprox_out/Float32/n1048576/normal    1.036–1.090   229074.8 -> 238445.8 ns   output offsets [64, 64, 64]
  some    enetw/prox_out/Float64/n32768/normal         0.972–1.082   8197.2 -> 8102.4 ns   output offsets [64, 64, 128]
  some    enetw/sparse_prox/Float32/n65536/tenth       0.881–1.078   20198.3 -> 19570.8 ns   output offsets [2944, 64, 128]
  some    enetw/prox_out/Float64/n65536/normal         0.997–1.075   18346.2 -> 18908.4 ns   output offsets [64, 64, 64]
  some    l1/prox_in/Float32/n1/normal                 1.000–1.056   4.9 -> 4.9 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n1/normal                1.000–1.056   4.9 -> 4.9 ns   output offsets [64, 64, 64]

A3c: 2 of 1000 cells slower in every process, 245 faster in every process, 18 above 1.05 in some processes only, 45 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 10.9 ns
  slower  l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns
  some    enet/prox_alloc/Float32/n1/normal            0.998–1.307   22.7 -> 22.9 ns   output offsets [64, 64, 64]
  some    enet/prox_alloc/Float32/n2/normal            0.986–1.294   23.4 -> 23.3 ns   output offsets [64, 64, 64]
  some    enet/prox_alloc/Float32/n4/normal            0.987–1.266   24.0 -> 24.0 ns   output offsets [48, 48, 48]
  some    enet/prox_alloc/Float32/n8/normal            0.986–1.257   25.0 -> 24.7 ns   output offsets [64, 64, 64]
  some    enet/prox_alloc/Float32/n32/normal           0.967–1.237   25.7 -> 25.2 ns   output offsets [160, 160, 480]
  some    enet/prox_alloc/Float32/n16/normal           0.976–1.196   25.7 -> 25.3 ns   output offsets [96, 96, 96]
  some    enetw/sparse_prox/Float32/n65536/tenth       0.907–1.195   20198.3 -> 21745.9 ns   output offsets [2944, 64, 128]
  some    enet/prox_alloc/Float32/n64/normal           0.947–1.165   32.8 -> 34.2 ns   output offsets [288, 288, 288]
  some    enet/prox_alloc/Float64/n32/normal           0.975–1.155   29.5 -> 28.8 ns   output offsets [288, 288, 288]
  some    l1w/prox_alloc/Float64/n64/normal            0.918–1.118   36.0 -> 32.7 ns   output offsets [544, 544, 544]
  some    l1/prox_out/Float64/n4/normal                1.000–1.100   5.5 -> 5.7 ns   output offsets [256, 64, 256]
  some    l1w/block_loop/Float32/n4/normal             1.000–1.091   6.0 -> 6.3 ns   output offsets [48, 48, 48]
  some    l1/prox_in/Float32/n1/normal                 1.000–1.056   4.9 -> 5.2 ns   output offsets [64, 64, 64]
  some    enetw/block_loop/Float64/n16/normal          0.977–1.054   11.8 -> 12.0 ns   output offsets [160, 160, 160]
  some    l1w/prox_in/Float32/n1/normal                1.000–1.053   5.2 -> 5.2 ns   output offsets [64, 64, 64]
  some    enet/prox_alloc/Float64/n4096/normal         0.923–1.050   1508.8 -> 1556.5 ns   output offsets [1408, 3584, 3392]
  some    l1/prox_g_out/Float64/n1/normal              1.000–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.000–1.050   5.5 -> 5.7 ns   output offsets [48, 48, 48]

A3cr: 6 of 1000 cells slower in every process, 240 faster in every process, 19 above 1.05 in some processes only, 40 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 11.2 ns
  slower  l1/prox_out/Float32/n1/normal                1.055–1.056   4.9 -> 5.2 ns
  slower  l1w/prox_in/Float32/n1/normal                1.053–1.053   5.2 -> 5.5 ns
  slower  l1w/prox_in/Float64/n1/normal                1.053–1.053   5.2 -> 5.5 ns
  slower  l1/prox_out/Float64/n4/normal                1.050–1.100   5.5 -> 5.7 ns
  slower  l1/prox_out/Float32/n4/normal                1.050–1.100   5.5 -> 5.7 ns
  some    enetw/sparse_prox/Float32/n65536/tenth       0.925–1.183   20198.3 -> 19765.6 ns   output offsets [2944, 64, 128]
  some    enet/prox_incopy/Float64/n4/normal           1.000–1.159   15.0 -> 15.0 ns   output offsets [256, 64, 256]
  some    l1w/prox_alloc/Float64/n64/normal            0.924–1.147   36.0 -> 31.3 ns   output offsets [544, 544, 544]
  some    enetw/prox_alloc/Float64/n32/normal          0.969–1.140   32.7 -> 32.2 ns   output offsets [288, 288, 288]
  some    enet/prox_g_out/Float64/n8/normal            0.999–1.136   20.1 -> 20.4 ns   output offsets [96, 96, 96]
  some    enet/valprox_out/Float32/n32/normal          1.002–1.135   21.2 -> 23.1 ns   output offsets [160, 160, 480]
  some    l1w/prox_alloc/Float32/n128/normal           0.878–1.096   36.6 -> 33.2 ns   output offsets [544, 544, 544]
  some    enetw/prox_incopy/Float32/n1/normal          0.999–1.094   15.6 -> 17.0 ns   output offsets [64, 64, 64]
  some    l1/prox_incopy/Float64/n1/normal             1.034–1.071   7.7 -> 8.2 ns   output offsets [64, 64, 64]
  some    enet/prox_alloc/Float64/n4096/normal         0.929–1.063   1508.8 -> 1496.5 ns   output offsets [1408, 3584, 3392]
  some    l1/prox_alloc/Float32/n4/normal              1.020–1.061   18.1 -> 18.8 ns   output offsets [48, 48, 48]
  some    l1/prox_alloc/Float32/n8/normal              1.023–1.056   19.2 -> 20.0 ns   output offsets [64, 64, 64]
  some    enetw/valprox_out/Float64/n1/normal          1.010–1.055   17.8 -> 18.1 ns   output offsets [64, 64, 64]
  some    l1w/prox_alloc/Float32/n64/normal            0.937–1.055   27.8 -> 26.2 ns   output offsets [288, 288, 288]
  some    l1/prox_alloc/Float32/n16/normal             0.973–1.053   20.2 -> 20.4 ns   output offsets [96, 96, 96]
  some    l1w/prox_out/Float64/n1/normal               1.000–1.053   5.2 -> 5.2 ns   output offsets [64, 64, 64]
  some    l1w/prox_out/Float32/n1/normal               1.000–1.053   5.2 -> 5.2 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float32/n1/normal              1.050–1.050   5.5 -> 5.7 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float64/n1/normal              1.000–1.050   5.5 -> 5.5 ns   output offsets [64, 64, 64]

A3i: 95 of 1000 cells slower in every process, 411 faster in every process, 37 above 1.05 in some processes only, 62 undecided (control unclean)
  slower  l1/prox_g_out/Float32/n32768/normal          1.542–1.635   37111.0 -> 57351.9 ns
  slower  l1/prox_g_out/Float64/n100000/normal         1.538–1.557   114898.2 -> 176988.0 ns
  slower  l1/prox_g_out/Float32/n100000/normal         1.537–1.558   114200.5 -> 175529.9 ns
  slower  l1/prox_g_out/Float64/n262144/normal         1.534–1.567   301550.7 -> 465189.5 ns
  slower  l1/prox_g_out/Float32/n8192/normal           1.533–1.622   9318.5 -> 14424.6 ns
  slower  l1/prox_g_out/Float32/n65536/normal          1.533–1.566   74691.1 -> 114840.5 ns
  slower  l1/prox_g_out/Float64/n8192/normal           1.532–1.700   9383.0 -> 14431.4 ns
  slower  l1/prox_g_out/Float64/n16384/normal          1.531–1.554   18773.7 -> 28985.8 ns
  slower  l1/prox_g_out/Float32/n4096/normal           1.529–1.545   4698.0 -> 7189.2 ns
  slower  l1/prox_g_out/Float64/n2048/normal           1.527–1.744   2323.1 -> 3559.0 ns
  slower  l1/prox_g_out/Float32/n262144/normal         1.526–1.554   300342.7 -> 459150.5 ns
  slower  l1/prox_g_out/Float32/n1048576/normal        1.523–1.566   1197458.0 -> 1835491.0 ns
  slower  l1/prox_g_out/Float32/n16384/normal          1.516–1.553   18767.7 -> 28682.0 ns
  slower  l1/prox_g_out/Float32/n2048/normal           1.509–1.561   2305.6 -> 3527.6 ns
  slower  l1/prox_g_out/Float64/n1024/normal           1.505–1.722   1164.3 -> 1756.5 ns
  slower  l1/prox_g_out/Float64/n1000/normal           1.504–1.726   1137.6 -> 1718.0 ns
  slower  l1/prox_g_out/Float32/n1024/normal           1.484–1.546   1159.5 -> 1758.7 ns
  slower  l1/prox_g_out/Float32/n1000/normal           1.471–4.269   1132.2 -> 1826.0 ns
  slower  l1/prox_g_out/Float64/n65536/normal          1.434–1.526   79624.8 -> 114828.8 ns
  slower  l1/prox_g_out/Float64/n32768/normal          1.420–1.564   37934.8 -> 57801.8 ns
  slower  enet/prox_g_out/Float32/n65536/normal        1.379–1.382   33590.4 -> 46367.1 ns
  slower  enet/prox_g_out/Float32/n262144/normal       1.379–1.381   133849.5 -> 184558.5 ns
  slower  enet/prox_g_out/Float32/n100000/normal       1.378–1.383   51021.5 -> 70332.9 ns
  slower  enet/prox_g_out/Float32/n1048576/normal      1.378–1.382   535581.0 -> 738530.3 ns
  slower  enet/prox_g_out/Float32/n16384/normal        1.377–1.383   8344.8 -> 11490.4 ns
  slower  enet/prox_g_out/Float32/n8192/normal         1.376–1.383   4180.6 -> 5755.7 ns
  slower  enet/prox_g_out/Float32/n32768/normal        1.376–1.384   16682.1 -> 22975.8 ns
  slower  enet/prox_g_out/Float32/n4096/normal         1.374–1.380   2103.7 -> 2891.9 ns
  slower  enet/prox_g_out/Float32/n2048/normal         1.374–1.382   1051.7 -> 1452.6 ns
  slower  enet/prox_g_out/Float32/n1000/normal         1.372–1.378   520.0 -> 713.1 ns
  slower  enet/prox_g_out/Float32/n1024/normal         1.371–1.379   530.1 -> 728.8 ns
  slower  l1/prox_g_out/Float64/n128/normal            1.355–1.612   152.6 -> 216.1 ns
  slower  enet/prox_g_out/Float32/n256/normal          1.344–1.352   139.1 -> 187.0 ns
  slower  l1/prox_g_out/Float64/n1048576/normal        1.324–1.548   1216839.0 -> 1868202.0 ns
  slower  l1/prox_g_out/Float32/n128/normal            1.317–1.450   151.6 -> 216.1 ns
  slower  enet/prox_g_out/Float32/n128/normal          1.307–1.321   76.0 -> 99.7 ns
  slower  enet/prox_g_out/Float64/n1048576/normal      1.265–1.271   1202683.0 -> 1521734.0 ns
  slower  enet/prox_g_out/Float64/n1024/normal         1.257–1.262   1170.9 -> 1472.0 ns
  slower  enet/prox_g_out/Float64/n1000/normal         1.256–1.262   1144.1 -> 1438.7 ns
  slower  enet/prox_g_out/Float64/n65536/normal        1.254–1.258   74697.4 -> 94012.9 ns
  some    enet/prox_out/Float32/n262144/normal         0.852–1.335   27812.0 -> 35870.8 ns   output offsets [64, 64, 64]
  some    enetw/prox_out/Float32/n65536/normal         1.002–1.295   8081.2 -> 9751.1 ns   output offsets [2944, 64, 128]
  some    enet/valprox_out/Float32/n65536/normal       0.804–1.282   9527.1 -> 11716.6 ns   output offsets [2944, 64, 128]
  some    enet/prox_out/Float64/n32768/normal          0.786–1.282   6997.3 -> 5808.9 ns   output offsets [64, 64, 128]
  some    enet/valprox_out/Float32/n1048576/normal     1.046–1.241   158977.1 -> 167042.2 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float32/n262144/normal      0.898–1.240   39074.6 -> 47166.0 ns   output offsets [64, 64, 64]
  some    enet/prox_out/Float64/n65536/normal          0.849–1.233   14826.3 -> 17207.7 ns   output offsets [64, 64, 64]
  some    enetw/valprox_out/Float32/n65536/normal      0.993–1.213   12756.3 -> 14211.0 ns   output offsets [2944, 64, 128]
  some    l1/prox_g_out/Float64/n8/normal              1.025–1.187   13.1 -> 13.7 ns   output offsets [96, 96, 96]
  some    l1/prox_g_out/Float32/n8/normal              1.023–1.187   13.1 -> 13.7 ns   output offsets [64, 64, 64]
  some    l1w/prox_out/Float32/n65536/normal           1.005–1.183   8432.9 -> 9131.9 ns   output offsets [2944, 64, 128]
  some    l1/prox_g_out/Float64/n4/normal              1.031–1.156   8.8 -> 9.3 ns   output offsets [256, 64, 256]
  some    l1/prox_g_out/Float32/n4/normal              1.031–1.156   8.8 -> 9.3 ns   output offsets [48, 48, 48]
  some    l1w/prox_alloc/Float64/n64/normal            0.923–1.153   36.0 -> 31.4 ns   output offsets [544, 544, 544]
  some    enetw/prox_out/Float32/n262144/normal        1.005–1.145   34677.7 -> 38771.1 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float32/n2/normal              1.042–1.126   6.6 -> 7.1 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float64/n2/normal              1.042–1.125   6.6 -> 7.1 ns   output offsets [48, 48, 48]
  some    l1w/valprox_out/Float32/n65536/normal        1.008–1.125   14014.8 -> 14707.5 ns   output offsets [2944, 64, 128]
  some    enet/valprox_out/Float64/n65536/normal       0.891–1.125   21017.1 -> 23210.3 ns   output offsets [64, 64, 64]
  some    l1/valprox_out/Float32/n65536/normal         0.801–1.124   9666.6 -> 9845.0 ns   output offsets [2944, 64, 128]
  some    l1w/prox_alloc/Float64/n16384/normal         0.938–1.098   5548.1 -> 6183.5 ns   output offsets [1728, 0, 2688]
  some    enetw/valprox_out/Float32/n262144/normal     1.001–1.096   57341.9 -> 62260.1 ns   output offsets [64, 64, 64]
  some    l1w/prox_out/Float64/n4/normal               1.000–1.087   6.3 -> 6.8 ns   output offsets [256, 64, 256]
  some    enet/prox_g_out/Float32/n2/normal            1.046–1.084   18.0 -> 18.9 ns   output offsets [64, 64, 64]
  some    enetw/sparse_prox/Float32/n65536/tenth       0.926–1.083   20198.3 -> 19788.2 ns   output offsets [2944, 64, 128]

Mi: 87 of 1000 cells slower in every process, 262 faster in every process, 31 above 1.05 in some processes only, 35 undecided (control unclean)
  slower  enet/prox_g_out/Float32/n65536/normal        1.380–1.382   33590.4 -> 46378.1 ns
  slower  enet/prox_g_out/Float32/n1048576/normal      1.378–1.382   535581.0 -> 738026.0 ns
  slower  enet/prox_g_out/Float32/n262144/normal       1.378–1.381   133849.5 -> 184668.7 ns
  slower  enet/prox_g_out/Float32/n100000/normal       1.378–1.382   51021.5 -> 70329.7 ns
  slower  enet/prox_g_out/Float32/n16384/normal        1.378–1.383   8344.8 -> 11498.0 ns
  slower  enet/prox_g_out/Float32/n8192/normal         1.377–1.382   4180.6 -> 5759.2 ns
  slower  enet/prox_g_out/Float32/n32768/normal        1.376–1.384   16682.1 -> 22989.2 ns
  slower  enet/prox_g_out/Float32/n4096/normal         1.375–1.380   2103.7 -> 2892.4 ns
  slower  enet/prox_g_out/Float32/n2048/normal         1.374–1.382   1051.7 -> 1453.3 ns
  slower  enet/prox_g_out/Float32/n1000/normal         1.373–1.379   520.0 -> 714.0 ns
  slower  enet/prox_g_out/Float32/n1024/normal         1.371–1.378   530.1 -> 729.7 ns
  slower  enet/prox_g_out/Float32/n256/normal          1.344–1.356   139.1 -> 187.4 ns
  slower  enet/prox_g_out/Float32/n128/normal          1.299–1.321   76.0 -> 99.6 ns
  slower  enet/prox_g_out/Float64/n1048576/normal      1.265–1.269   1202683.0 -> 1520932.0 ns
  slower  enet/prox_g_out/Float64/n1000/normal         1.255–1.264   1144.1 -> 1438.3 ns
  slower  enet/prox_g_out/Float64/n1024/normal         1.254–1.264   1170.9 -> 1472.4 ns
  slower  enet/prox_g_out/Float64/n2048/normal         1.250–1.259   2343.3 -> 2935.1 ns
  slower  enet/prox_g_out/Float64/n32768/normal        1.250–1.260   37382.1 -> 46897.7 ns
  slower  enet/prox_g_out/Float64/n4096/normal         1.249–1.258   4672.1 -> 5856.0 ns
  slower  enet/prox_g_out/Float64/n8192/normal         1.249–1.258   9315.3 -> 11669.6 ns
  slower  enet/prox_g_out/Float64/n65536/normal        1.249–1.262   74697.4 -> 93841.1 ns
  slower  enet/prox_g_out/Float64/n256/normal          1.248–1.258   299.0 -> 374.6 ns
  slower  enet/prox_g_out/Float64/n262144/normal       1.248–1.259   298619.6 -> 374666.2 ns
  slower  enet/prox_g_out/Float64/n16384/normal        1.247–1.258   18616.0 -> 23340.6 ns
  slower  enet/prox_g_out/Float64/n100000/normal       1.246–1.264   114015.3 -> 143014.4 ns
  slower  enet/prox_g_out/Float64/n128/normal          1.238–1.250   155.2 -> 192.8 ns
  slower  l1/prox_g_out/Float32/n8192/normal           1.229–1.472   9318.5 -> 11743.3 ns
  slower  l1/prox_g_out/Float32/n32768/normal          1.223–1.485   37111.0 -> 53643.0 ns
  slower  l1/prox_g_out/Float32/n1000/normal           1.221–3.336   1132.2 -> 1667.7 ns
  slower  l1/prox_g_out/Float32/n65536/normal          1.220–1.447   74691.1 -> 107273.2 ns
  slower  l1/prox_g_out/Float32/n100000/normal         1.220–1.454   114200.5 -> 164500.5 ns
  slower  l1/prox_g_out/Float32/n1048576/normal        1.218–1.440   1197458.0 -> 1701309.0 ns
  slower  l1/prox_g_out/Float32/n262144/normal         1.217–1.457   300342.7 -> 434239.6 ns
  slower  enet/prox_g_out/Float64/n64/normal           1.214–1.229   83.9 -> 102.5 ns
  slower  l1/prox_g_out/Float32/n16384/normal          1.213–1.427   18767.7 -> 26346.2 ns
  slower  l1/prox_g_out/Float64/n8192/normal           1.213–1.242   9383.0 -> 11397.6 ns
  slower  l1/prox_g_out/Float64/n1048576/normal        1.206–1.243   1216839.0 -> 1479325.0 ns
  slower  l1/prox_g_out/Float64/n262144/normal         1.205–1.440   301550.7 -> 367937.5 ns
  slower  l1/prox_g_out/Float64/n100000/normal         1.202–1.464   114898.2 -> 139896.5 ns
  slower  l1/prox_g_out/Float64/n2048/normal           1.195–1.222   2323.1 -> 2809.2 ns
  some    l1/prox_g_out/Float32/n64/normal             1.036–1.220   80.5 -> 96.8 ns   output offsets [288, 288, 288]
  some    l1w/prox_alloc/Float32/n128/normal           0.983–1.217   36.6 -> 37.2 ns   output offsets [544, 544, 544]
  some    l1/prox_g_out/Float32/n32/normal             1.033–1.213   39.6 -> 42.0 ns   output offsets [160, 160, 480]
  some    l1/prox_g_out/Float64/n32/normal             1.021–1.209   39.9 -> 42.2 ns   output offsets [288, 288, 288]
  some    l1/prox_g_out/Float64/n16/normal             1.008–1.200   21.9 -> 22.2 ns   output offsets [160, 160, 160]
  some    l1/prox_g_out/Float32/n8/normal              1.021–1.187   13.1 -> 13.4 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float64/n8/normal              1.001–1.167   13.1 -> 13.4 ns   output offsets [96, 96, 96]
  some    l1w/prox_alloc/Float64/n32/normal            1.018–1.161   26.3 -> 29.0 ns   output offsets [288, 288, 288]
  some    l1/prox_g_out/Float32/n4/normal              1.031–1.156   8.8 -> 9.0 ns   output offsets [48, 48, 48]
  some    l1/prox_g_out/Float32/n2/normal              1.042–1.125   6.6 -> 6.8 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float64/n4/normal              1.000–1.125   8.8 -> 9.0 ns   output offsets [256, 64, 256]
  some    l1/prox_out/Float32/n1/normal                1.000–1.111   4.9 -> 4.9 ns   output offsets [64, 64, 64]
  some    enetw/sparse_prox/Float32/n65536/tenth       0.986–1.111   20198.3 -> 20254.8 ns   output offsets [2944, 64, 128]
  some    l1w/prox_alloc/Float64/n64/normal            0.994–1.106   36.0 -> 34.0 ns   output offsets [544, 544, 544]
  some    l1/prox_out/Float32/n2/normal                1.000–1.100   5.5 -> 5.5 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float32/n65536/normal       0.999–1.099   9527.1 -> 9513.6 ns   output offsets [2944, 64, 128]
  some    l1/prox_g_out/Float64/n2/normal              1.000–1.083   6.6 -> 6.8 ns   output offsets [48, 48, 48]
  some    enet/prox_g_out/Float32/n2/normal            1.046–1.083   18.0 -> 19.3 ns   output offsets [64, 64, 64]
  some    l1/prox_alloc/Float32/n16/normal             1.013–1.075   20.2 -> 20.5 ns   output offsets [96, 96, 96]
  some    enet/prox_incopy/Float32/n1/normal           0.998–1.067   13.9 -> 14.5 ns   output offsets [64, 64, 64]
  some    enet/prox_g_out/Float32/n8/normal            1.016–1.066   19.0 -> 19.7 ns   output offsets [64, 64, 64]
  some    enet/prox_incopy/Float32/n2/normal           1.004–1.058   14.3 -> 14.8 ns   output offsets [64, 64, 64]
  some    enet/prox_g_out/Float64/n2/normal            1.046–1.057   17.3 -> 18.2 ns   output offsets [48, 48, 48]
  some    l1/prox_in/Float32/n1/normal                 1.000–1.056   4.9 -> 5.2 ns   output offsets [64, 64, 64]
  some    l1w/valprox_out/Float64/n16/normal           1.010–1.053   16.7 -> 17.0 ns   output offsets [160, 160, 160]

## epyc7513b-julia1.13.4 (3 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 5000 of 5000

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3c | A3cr | A3i | Mi | control |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 0.99 [0.94–1.03] | 0.99 [0.95–1.03] | 0.99 [0.95–1.03] | 0.99 [0.92–1.03] | 1.00 [0.97–1.03] | 1.00 |
| enet | block_loop | medium | 4 | 0.74 [0.68–0.79] | 0.73 [0.68–0.79] | 0.73 [0.68–0.79] | 0.74 [0.68–0.82] | 1.00 [1.00–1.01] | 1.00 |
| enet | prox_alloc | short | 14 | 0.98 [0.86–1.00] | 0.98 [0.86–1.01] | 0.98 [0.88–1.01] | 1.00 [0.98–1.02] | 1.02 [1.00–1.12] | 1.00 |
| enet | prox_alloc | medium | 14 | 0.94 [0.90–1.05] | 0.90 [0.70–1.01] | 0.94 [0.90–1.00] | 0.92 [0.72–1.00] | 0.99 [0.84–1.07] | 0.99 |
| enet | prox_alloc | large | 12 | 0.91 [0.81–1.01] | 1.05 [0.94–1.25] | 0.94 [0.75–1.02] | 0.97 [0.77–1.11] | 1.02 [0.96–1.20] | 1.08 |
| enet | prox_g_out | short | 14 | 1.00 [0.97–1.02] | 1.00 [0.97–1.10] | 1.00 [0.95–1.09] | 1.10 [1.01–1.28] | 1.10 [1.00–1.27] | 1.01 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.30 [1.24–1.38] | 1.31 [1.24–1.38] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.32 [1.26–1.38] | 1.32 [1.26–1.38] | 1.00 |
| enet | prox_in | short | 14 | 0.86 [0.72–1.00] | 0.88 [0.72–1.03] | 0.88 [0.72–1.03] | 0.64 [0.22–1.06] | 0.66 [0.23–1.09] | 1.00 |
| enet | prox_in | medium | 14 | 0.70 [0.69–0.71] | 0.74 [0.69–1.00] | 0.73 [0.69–1.00] | 0.13 [0.09–0.23] | 0.19 [0.13–0.29] | 1.00 |
| enet | prox_in | large | 12 | 0.69 [0.69–0.69] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 0.12 [0.08–0.17] | 0.18 [0.12–0.25] | 1.00 |
| enet | prox_incopy | short | 14 | 0.88 [0.74–1.00] | 0.89 [0.74–1.02] | 0.89 [0.74–1.02] | 0.69 [0.28–1.06] | 0.71 [0.29–1.06] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.72 [0.71–0.74] | 0.75 [0.71–1.00] | 0.75 [0.71–1.00] | 0.19 [0.12–0.30] | 0.24 [0.16–0.36] | 1.00 |
| enet | prox_incopy | large | 12 | 0.73 [0.72–0.75] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 0.22 [0.15–0.32] | 0.27 [0.19–0.39] | 1.00 |
| enet | prox_out | short | 14 | 0.96 [0.74–1.00] | 0.97 [0.73–1.02] | 0.97 [0.75–1.03] | 1.00 [0.75–1.06] | 1.03 [1.00–1.09] | 1.00 |
| enet | prox_out | medium | 14 | 0.72 [0.68–0.85] | 0.75 [0.68–1.00] | 0.75 [0.68–1.00] | 0.72 [0.68–0.83] | 1.01 [0.99–1.04] | 1.00 |
| enet | prox_out | large | 12 | 0.85 [0.69–1.33] | 1.00 [1.00–1.01] | 1.00 [0.99–1.01] | 0.86 [0.70–1.33] | 1.00 [1.00–1.02] | 1.00 |
| enet | sparse_prox | large | 2 | 1.10 [1.09–1.11] | 1.10 [1.09–1.10] | 1.10 [1.09–1.10] | 1.10 [1.10–1.11] | 1.00 [1.00–1.01] | 0.97 |
| enet | valprox_out | short | 14 | 0.97 [0.88–1.01] | 0.98 [0.88–1.02] | 0.98 [0.89–1.04] | 0.98 [0.87–1.04] | 1.00 [0.97–1.02] | 1.00 |
| enet | valprox_out | medium | 14 | 0.79 [0.76–0.90] | 0.82 [0.76–1.00] | 0.82 [0.76–1.00] | 0.79 [0.76–0.91] | 1.00 [1.00–1.01] | 1.00 |
| enet | valprox_out | large | 12 | 0.89 [0.77–1.21] | 1.00 [0.99–1.00] | 1.00 [0.99–1.00] | 0.90 [0.77–1.24] | 1.00 [1.00–1.01] | 1.00 |
| enetw | block_loop | short | 4 | 1.01 [0.90–1.14] | 1.02 [0.95–1.14] | 1.01 [0.90–1.11] | 1.01 [0.93–1.11] | 1.01 [1.00–1.03] | 1.00 |
| enetw | block_loop | medium | 4 | 0.88 [0.86–0.90] | 0.89 [0.86–0.94] | 0.94 [0.86–1.03] | 0.88 [0.86–0.91] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.98 [0.93–1.01] | 0.98 [0.91–1.02] | 0.98 [0.90–1.02] | 1.00 [0.97–1.05] | 1.01 [1.00–1.02] | 1.00 |
| enetw | prox_alloc | medium | 14 | 0.94 [0.74–1.04] | 0.95 [0.73–1.03] | 0.98 [0.89–1.11] | 0.93 [0.74–1.04] | 1.00 [0.88–1.06] | 0.99 |
| enetw | prox_alloc | large | 12 | 1.06 [1.00–1.30] | 1.03 [0.96–1.32] | 0.97 [0.77–1.10] | 1.03 [0.89–1.27] | 1.05 [0.97–1.33] | 1.05 |
| enetw | prox_g_out | short | 14 | 1.00 [0.99–1.02] | 1.00 [0.98–1.01] | 1.00 [0.99–1.02] | 1.06 [1.00–1.15] | 1.06 [1.00–1.14] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [0.99–1.01] | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 1.15 [1.14–1.15] | 1.15 [1.13–1.15] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 1.00 [1.00–1.01] | 1.15 [1.15–1.15] | 1.15 [1.15–1.15] | 1.00 |
| enetw | prox_in | short | 14 | 0.92 [0.82–1.02] | 0.92 [0.81–1.05] | 0.92 [0.81–1.05] | 0.64 [0.24–1.05] | 0.66 [0.28–1.05] | 1.00 |
| enetw | prox_in | medium | 14 | 0.78 [0.77–0.80] | 0.81 [0.78–1.00] | 0.81 [0.78–1.00] | 0.15 [0.10–0.27] | 0.20 [0.13–0.32] | 1.00 |
| enetw | prox_in | large | 12 | 0.78 [0.78–0.82] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 0.15 [0.10–0.25] | 0.19 [0.13–0.29] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.92 [0.82–1.02] | 0.93 [0.82–1.04] | 0.93 [0.83–1.04] | 0.69 [0.30–1.04] | 0.70 [0.32–1.04] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.80 [0.78–0.82] | 0.83 [0.79–1.00] | 0.83 [0.79–1.00] | 0.21 [0.13–0.32] | 0.26 [0.17–0.37] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.81 [0.79–0.84] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 0.25 [0.16–0.37] | 0.28 [0.19–0.40] | 1.00 |
| enetw | prox_out | short | 14 | 0.98 [0.88–1.02] | 1.00 [0.89–1.07] | 0.99 [0.90–1.05] | 1.00 [0.88–1.05] | 1.02 [1.00–1.04] | 1.00 |
| enetw | prox_out | medium | 14 | 0.83 [0.73–0.98] | 0.84 [0.73–1.00] | 0.84 [0.73–1.01] | 0.84 [0.73–0.98] | 1.00 [1.00–1.03] | 1.00 |
| enetw | prox_out | large | 12 | 1.02 [0.95–1.14] | 1.00 [1.00–1.00] | 1.00 [1.00–1.01] | 1.01 [0.95–1.12] | 1.00 [1.00–1.01] | 1.00 |
| enetw | sparse_prox | large | 2 | 1.02 [0.98–1.07] | 1.01 [0.95–1.07] | 1.00 [0.94–1.07] | 1.00 [0.93–1.08] | 1.00 [0.99–1.00] | 1.04 |
| enetw | valprox_out | short | 14 | 0.96 [0.82–1.02] | 0.97 [0.83–1.02] | 0.97 [0.85–1.02] | 0.97 [0.84–1.03] | 1.01 [1.00–1.03] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.89 [0.83–0.99] | 0.90 [0.83–1.00] | 0.90 [0.83–1.01] | 0.90 [0.83–0.99] | 1.00 [1.00–1.01] | 1.00 |
| enetw | valprox_out | large | 12 | 1.01 [0.97–1.10] | 1.00 [0.99–1.00] | 1.00 [1.00–1.01] | 1.01 [0.97–1.08] | 1.00 [1.00–1.01] | 1.00 |
| l1 | block_loop | short | 4 | 0.98 [0.96–1.00] | 1.01 [0.96–1.05] | 0.99 [0.96–1.05] | 1.00 [0.96–1.04] | 1.00 [1.00–1.00] | 1.01 |
| l1 | block_loop | medium | 4 | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 0.69 [0.67–0.75] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.98 [0.94–1.01] | 1.01 [0.96–1.04] | 1.03 [0.98–1.05] | 0.99 [0.98–1.01] | 1.00 [1.00–1.02] | 1.01 |
| l1 | prox_alloc | medium | 14 | 0.94 [0.84–1.01] | 0.95 [0.89–1.00] | 0.98 [0.92–1.09] | 0.95 [0.85–1.08] | 0.99 [0.80–1.20] | 0.98 |
| l1 | prox_alloc | large | 12 | 1.02 [0.92–1.09] | 0.97 [0.79–1.02] | 0.93 [0.71–1.02] | 1.02 [0.92–1.07] | 0.99 [0.98–1.01] | 1.04 |
| l1 | prox_g_out | short | 14 | 1.00 [0.93–1.05] | 1.00 [0.93–1.05] | 1.00 [0.92–1.05] | 1.11 [1.02–1.31] | 1.09 [1.01–1.21] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.98 [0.97–0.99] | 0.98 [0.97–1.00] | 0.98 [0.97–0.99] | 1.50 [1.41–1.55] | 1.20 [1.09–1.35] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.98 [0.97–0.98] | 0.98 [0.97–0.98] | 0.98 [0.97–0.98] | 1.53 [1.52–1.54] | 1.32 [1.21–1.47] | 1.00 |
| l1 | prox_in | short | 14 | 0.84 [0.68–1.00] | 0.84 [0.68–1.00] | 0.85 [0.68–1.06] | 0.55 [0.15–1.06] | 0.57 [0.17–1.06] | 1.01 |
| l1 | prox_in | medium | 14 | 0.68 [0.68–0.69] | 0.72 [0.68–1.00] | 0.72 [0.68–1.00] | 0.12 [0.08–0.17] | 0.18 [0.12–0.25] | 1.00 |
| l1 | prox_in | large | 12 | 0.68 [0.68–0.68] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 0.12 [0.08–0.17] | 0.18 [0.12–0.25] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.88 [0.70–1.03] | 0.90 [0.72–1.12] | 0.92 [0.72–1.16] | 0.63 [0.22–1.04] | 0.65 [0.24–1.06] | 1.00 |
| l1 | prox_incopy | medium | 14 | 0.71 [0.70–0.73] | 0.74 [0.70–1.00] | 0.74 [0.70–1.00] | 0.18 [0.12–0.29] | 0.23 [0.16–0.36] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.71–0.74] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 0.22 [0.15–0.32] | 0.27 [0.19–0.39] | 1.00 |
| l1 | prox_out | short | 14 | 0.96 [0.79–1.05] | 0.98 [0.77–1.10] | 1.01 [0.79–1.11] | 0.97 [0.79–1.05] | 1.04 [1.00–1.11] | 1.01 |
| l1 | prox_out | medium | 14 | 0.68 [0.67–0.77] | 0.72 [0.67–1.00] | 0.72 [0.67–1.00] | 0.68 [0.67–0.77] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_out | large | 12 | 0.79 [0.67–1.00] | 1.00 [0.99–1.01] | 1.00 [1.00–1.01] | 0.79 [0.67–0.99] | 1.00 [0.99–1.02] | 1.00 |
| l1 | sparse_prox | large | 2 | 0.99 [0.98–0.99] | 0.97 [0.96–0.99] | 0.97 [0.95–1.00] | 1.00 [0.99–1.01] | 0.98 [0.97–0.99] | 1.00 |
| l1 | valprox_out | short | 14 | 0.96 [0.86–1.03] | 0.97 [0.86–1.03] | 1.00 [0.86–1.08] | 0.96 [0.87–1.03] | 1.01 [1.00–1.04] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.77 [0.74–0.86] | 0.80 [0.74–1.00] | 0.80 [0.74–1.00] | 0.77 [0.74–0.88] | 1.00 [1.00–1.00] | 1.00 |
| l1 | valprox_out | large | 12 | 0.84 [0.75–0.97] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 0.84 [0.75–0.98] | 1.00 [0.99–1.01] | 1.00 |
| l1w | block_loop | short | 4 | 0.98 [0.93–1.00] | 1.01 [0.96–1.05] | 0.99 [0.93–1.05] | 1.01 [0.93–1.09] | 1.00 [1.00–1.00] | 1.02 |
| l1w | block_loop | medium | 4 | 0.72 [0.68–0.76] | 0.73 [0.67–0.79] | 0.73 [0.67–0.79] | 0.73 [0.68–0.78] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 0.97 [0.89–1.00] | 0.99 [0.92–1.03] | 0.99 [0.93–1.02] | 0.99 [0.91–1.02] | 1.01 [0.99–1.02] | 1.00 |
| l1w | prox_alloc | medium | 14 | 0.97 [0.84–1.10] | 0.98 [0.88–1.10] | 0.98 [0.88–1.10] | 0.99 [0.88–1.08] | 0.98 [0.81–1.02] | 1.02 |
| l1w | prox_alloc | large | 12 | 1.10 [1.04–1.38] | 1.00 [0.96–1.12] | 0.95 [0.78–1.01] | 1.10 [1.04–1.25] | 1.00 [0.99–1.01] | 1.07 |
| l1w | prox_g_out | short | 14 | 0.99 [0.91–1.04] | 1.02 [0.91–1.10] | 0.99 [0.91–1.04] | 1.08 [1.04–1.11] | 1.03 [1.00–1.10] | 1.03 |
| l1w | prox_g_out | medium | 14 | 0.96 [0.96–0.97] | 1.03 [0.96–1.11] | 0.96 [0.96–0.97] | 1.10 [1.08–1.11] | 1.05 [1.00–1.11] | 1.05 |
| l1w | prox_g_out | large | 12 | 0.96 [0.96–0.96] | 1.03 [0.96–1.10] | 0.96 [0.96–0.96] | 1.09 [1.03–1.10] | 1.04 [1.00–1.10] | 1.05 |
| l1w | prox_in | short | 14 | 0.89 [0.75–1.00] | 0.90 [0.77–1.00] | 0.92 [0.76–1.05] | 0.54 [0.16–1.05] | 0.56 [0.19–1.05] | 1.00 |
| l1w | prox_in | medium | 14 | 0.71 [0.71–0.73] | 0.75 [0.71–1.00] | 0.75 [0.71–1.00] | 0.13 [0.09–0.19] | 0.19 [0.13–0.26] | 1.00 |
| l1w | prox_in | large | 12 | 0.71 [0.71–0.71] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 0.15 [0.09–0.30] | 0.18 [0.13–0.27] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.91 [0.77–1.02] | 0.94 [0.78–1.12] | 0.95 [0.78–1.12] | 0.62 [0.23–1.03] | 0.64 [0.27–1.03] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.74 [0.72–0.76] | 0.77 [0.72–1.00] | 0.77 [0.72–1.00] | 0.20 [0.13–0.31] | 0.25 [0.17–0.36] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.75 [0.73–0.77] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 0.25 [0.16–0.42] | 0.28 [0.19–0.39] | 1.00 |
| l1w | prox_out | short | 14 | 0.97 [0.84–1.00] | 0.96 [0.82–1.00] | 0.98 [0.84–1.04] | 0.98 [0.84–1.05] | 1.03 [1.00–1.05] | 1.00 |
| l1w | prox_out | medium | 14 | 0.79 [0.66–0.99] | 0.79 [0.65–1.00] | 0.79 [0.67–1.00] | 0.79 [0.67–0.99] | 1.00 [1.00–1.01] | 1.00 |
| l1w | prox_out | large | 12 | 1.01 [0.99–1.03] | 1.00 [1.00–1.01] | 1.00 [0.99–1.01] | 1.01 [0.99–1.05] | 1.00 [1.00–1.01] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.01 [1.00–1.02] | 1.01 [1.01–1.01] | 1.02 [1.02–1.03] | 1.02 [1.01–1.02] | 1.00 [1.00–1.00] | 1.00 |
| l1w | valprox_out | short | 14 | 0.96 [0.88–1.00] | 0.98 [0.89–1.03] | 0.99 [0.90–1.04] | 0.98 [0.90–1.03] | 1.01 [0.99–1.03] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.89 [0.82–0.99] | 0.89 [0.83–1.00] | 0.89 [0.83–1.00] | 0.89 [0.83–0.99] | 1.00 [1.00–1.01] | 1.00 |
| l1w | valprox_out | large | 12 | 1.00 [0.99–1.03] | 1.00 [0.99–1.00] | 1.00 [0.99–1.00] | 1.00 [0.99–1.03] | 1.00 [1.00–1.00] | 1.00 |

A3M4: 8 of 1000 cells slower in every process, 396 faster in every process, 21 above 1.05 in some processes only, 74 undecided (control unclean)
  slower  enet/prox_out/Float32/n262144/normal         1.283–1.338   27854.1 -> 37057.2 ns
  slower  enet/valprox_out/Float32/n262144/normal      1.198–1.271   39077.7 -> 47040.8 ns
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 11.2 ns
  slower  l1w/prox_alloc/Float64/n8192/normal          1.076–1.097   2490.2 -> 2961.7 ns
  slower  enetw/prox_out/Float32/n262144/normal        1.071–1.276   34590.6 -> 39378.0 ns
  slower  enetw/sparse_prox/Float64/n65536/tenth       1.065–1.071   21386.6 -> 22791.9 ns
  slower  enetw/valprox_out/Float32/n262144/normal     1.064–1.170   57123.3 -> 62800.8 ns
  slower  enetw/valprox_out/Float32/n1048576/normal    1.060–1.178   229170.6 -> 247442.5 ns
  some    enet/prox_out/Float64/n65536/normal          1.026–1.349   14476.5 -> 15773.9 ns   output offsets [64, 256, 128]
  some    enetw/prox_out/Float32/n1048576/normal       1.027–1.269   138501.0 -> 152418.7 ns   output offsets [64, 64, 128]
  some    enet/valprox_out/Float64/n65536/normal       1.019–1.252   20471.5 -> 23615.6 ns   output offsets [64, 256, 128]
  some    enet/prox_out/Float64/n32768/normal          0.707–1.245   6859.6 -> 5126.4 ns   output offsets [640, 3328, 64]
  some    enetw/prox_out/Float64/n32768/normal         0.994–1.202   8019.9 -> 8375.7 ns   output offsets [640, 3328, 64]
  some    enet/valprox_out/Float64/n32768/normal       0.789–1.187   9337.7 -> 7495.0 ns   output offsets [640, 3328, 64]
  some    enetw/prox_out/Float64/n65536/normal         1.005–1.176   18708.3 -> 20516.4 ns   output offsets [64, 256, 128]
  some    l1/valprox_out/Float64/n65536/normal         0.889–1.162   20022.9 -> 20946.7 ns   output offsets [64, 256, 128]
  some    l1w/prox_out/Float32/n1048576/normal         1.024–1.123   139964.9 -> 143454.6 ns   output offsets [64, 64, 128]
  some    enetw/valprox_out/Float64/n65536/normal      1.016–1.107   30116.6 -> 32361.8 ns   output offsets [64, 256, 128]
  some    enetw/valprox_out/Float64/n32768/normal      0.997–1.081   13145.0 -> 13087.2 ns   output offsets [640, 3328, 64]
  some    enetw/block_loop/Float64/n4/normal           1.000–1.081   9.6 -> 9.8 ns   output offsets [64, 64, 64]
  some    l1w/valprox_out/Float64/n4/normal            0.974–1.076   12.6 -> 12.3 ns   output offsets [64, 64, 64]
  some    l1w/prox_out/Float64/n32768/normal           0.997–1.074   8283.6 -> 8321.9 ns   output offsets [640, 3328, 64]
  some    l1w/prox_out/Float32/n262144/normal          1.019–1.072   35209.2 -> 35888.3 ns   output offsets [64, 64, 128]
  some    l1w/valprox_out/Float32/n1048576/normal      1.018–1.069   231545.0 -> 236661.0 ns   output offsets [64, 64, 128]
  some    l1/prox_alloc/Float32/n100000/normal         1.014–1.057   16224.4 -> 16916.4 ns   output offsets [2752, 2752, 2816]
  some    enetw/valprox_out/Float64/n1/normal          1.004–1.053   17.7 -> 17.8 ns   output offsets [64, 64, 64]
  some    l1w/valprox_out/Float32/n262144/normal       1.019–1.053   58267.4 -> 59810.0 ns   output offsets [64, 64, 128]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.2 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.000–1.050   5.5 -> 5.5 ns   output offsets [48, 48, 48]

A3c: 3 of 1000 cells slower in every process, 238 faster in every process, 16 above 1.05 in some processes only, 57 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 11.2 ns
  slower  l1w/prox_incopy/Float32/n1/normal            1.088–1.147   9.3 -> 10.4 ns
  slower  l1/prox_out/Float64/n4/normal                1.050–1.100   5.5 -> 5.7 ns
  some    enetw/prox_out/Float64/n1/normal             1.023–1.690   11.5 -> 12.3 ns   output offsets [64, 64, 64]
  some    enetw/prox_out/Float64/n8/normal             1.000–1.487   13.4 -> 13.7 ns   output offsets [96, 96, 96]
  some    enetw/prox_out/Float64/n16/normal            0.998–1.403   14.6 -> 14.8 ns   output offsets [160, 160, 160]
  some    enetw/prox_out/Float64/n32/normal            0.952–1.277   17.2 -> 16.7 ns   output offsets [288, 288, 288]
  some    enetw/valprox_out/Float64/n4/normal          1.001–1.229   21.0 -> 21.3 ns   output offsets [64, 64, 64]
  some    enetw/prox_incopy/Float64/n4/normal          0.985–1.107   17.5 -> 17.5 ns   output offsets [64, 64, 64]
  some    l1w/valprox_out/Float64/n4/normal            0.980–1.105   12.6 -> 12.6 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.050–1.100   5.5 -> 5.7 ns   output offsets [48, 48, 48]
  some    enetw/block_loop/Float64/n4/normal           0.972–1.083   9.6 -> 9.6 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float32/n16/normal          0.957–1.070   20.2 -> 19.7 ns   output offsets [96, 192, 96]
  some    enetw/prox_in/Float64/n8/normal              0.968–1.068   17.0 -> 16.4 ns   output offsets [96, 96, 96]
  some    enetw/sparse_prox/Float64/n65536/tenth       0.940–1.065   21386.6 -> 20641.3 ns   output offsets [64, 256, 128]
  some    l1w/prox_in/Float64/n1/normal                1.000–1.053   5.2 -> 5.2 ns   output offsets [64, 64, 64]
  some    l1/block_loop/Float64/n4/normal              1.000–1.053   5.2 -> 5.5 ns   output offsets [64, 64, 64]
  some    l1w/prox_in/Float32/n1/normal                0.999–1.053   5.2 -> 5.2 ns   output offsets [64, 64, 64]
  some    enetw/prox_out/Float64/n65536/normal         0.999–1.051   18708.3 -> 18855.9 ns   output offsets [64, 256, 128]

A3cr: 5 of 1000 cells slower in every process, 232 faster in every process, 25 above 1.05 in some processes only, 46 undecided (control unclean)
  slower  enetw/block_loop/Float32/n4/normal           1.111–1.139   9.8 -> 10.9 ns
  slower  l1w/prox_incopy/Float32/n1/normal            1.088–1.118   9.3 -> 10.4 ns
  slower  l1/prox_out/Float32/n1/normal                1.056–1.111   4.9 -> 5.5 ns
  slower  l1w/prox_in/Float64/n1/normal                1.053–1.053   5.2 -> 5.5 ns
  slower  l1/block_loop/Float64/n4/normal              1.053–1.053   5.2 -> 5.5 ns
  some    enetw/prox_out/Float64/n1/normal             1.047–1.690   11.5 -> 12.3 ns   output offsets [64, 64, 64]
  some    enetw/prox_out/Float64/n8/normal             1.000–1.488   13.4 -> 13.7 ns   output offsets [96, 96, 96]
  some    enetw/prox_out/Float64/n16/normal            0.981–1.404   14.6 -> 14.5 ns   output offsets [160, 160, 160]
  some    l1/prox_alloc/Float32/n100000/normal         0.975–1.299   16224.4 -> 16098.4 ns   output offsets [2752, 2752, 2816]
  some    enetw/prox_out/Float64/n32/normal            0.938–1.269   17.2 -> 16.7 ns   output offsets [288, 288, 288]
  some    enetw/valprox_out/Float64/n4/normal          1.003–1.218   21.0 -> 21.5 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float32/n16/normal          0.980–1.114   20.2 -> 19.8 ns   output offsets [96, 192, 96]
  some    enetw/prox_alloc/Float32/n256/normal         0.865–1.113   62.8 -> 57.6 ns   output offsets [1088, 1088, 1152]
  some    l1w/valprox_out/Float64/n4/normal            1.007–1.104   12.6 -> 12.9 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float32/n8/normal           0.985–1.101   17.7 -> 18.1 ns   output offsets [320, 1280, 64]
  some    l1/prox_out/Float32/n2/normal                1.000–1.100   5.5 -> 6.0 ns   output offsets [64, 64, 64]
  some    enetw/prox_alloc/Float64/n4096/normal        1.022–1.083   1528.3 -> 1618.1 ns   output offsets [3840, 64, 2496]
  some    enetw/prox_incopy/Float64/n4/normal          0.985–1.082   17.5 -> 17.5 ns   output offsets [64, 64, 64]
  some    enetw/sparse_prox/Float64/n65536/tenth       0.921–1.075   21386.6 -> 20541.9 ns   output offsets [64, 256, 128]
  some    enet/valprox_out/Float32/n32/normal          0.968–1.070   21.3 -> 20.8 ns   output offsets [160, 160, 3360]
  some    enet/valprox_out/Float32/n4/normal           1.019–1.065   16.4 -> 16.8 ns   output offsets [48, 48, 48]
  some    l1/valprox_out/Float32/n1/normal             1.028–1.057   9.6 -> 10.1 ns   output offsets [64, 64, 64]
  some    l1/prox_in/Float32/n1/normal                 1.000–1.055   4.9 -> 5.2 ns   output offsets [64, 64, 64]
  some    l1/prox_alloc/Float32/n4/normal              1.032–1.053   18.2 -> 19.3 ns   output offsets [48, 48, 48]
  some    l1w/prox_in/Float32/n1/normal                1.042–1.053   5.2 -> 5.5 ns   output offsets [64, 64, 64]
  some    l1w/prox_out/Float64/n1/normal               1.000–1.053   5.2 -> 5.2 ns   output offsets [64, 64, 64]
  some    l1/valprox_out/Float32/n2/normal             1.000–1.053   10.4 -> 10.9 ns   output offsets [64, 64, 64]
  some    l1/valprox_out/Float64/n4/normal             0.999–1.051   10.7 -> 11.2 ns   output offsets [64, 64, 64]
  some    enetw/block_loop/Float64/n4/normal           0.972–1.050   9.6 -> 9.8 ns   output offsets [64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.050–1.050   5.5 -> 5.7 ns   output offsets [48, 48, 48]

A3i: 99 of 1000 cells slower in every process, 407 faster in every process, 33 above 1.05 in some processes only, 71 undecided (control unclean)
  slower  l1/prox_g_out/Float32/n65536/normal          1.536–1.540   74521.6 -> 114908.7 ns
  slower  l1/prox_g_out/Float64/n100000/normal         1.530–1.543   115208.3 -> 177454.4 ns
  slower  l1/prox_g_out/Float32/n32768/normal          1.529–1.621   37279.0 -> 57163.4 ns
  slower  l1/prox_g_out/Float32/n100000/normal         1.528–1.535   114099.8 -> 175058.1 ns
  slower  l1/prox_g_out/Float64/n262144/normal         1.527–1.537   301363.3 -> 462993.0 ns
  slower  l1/prox_g_out/Float32/n262144/normal         1.522–1.538   299861.9 -> 457455.0 ns
  slower  l1/prox_g_out/Float64/n8192/normal           1.522–1.537   9394.2 -> 14345.3 ns
  slower  l1/prox_g_out/Float32/n4096/normal           1.517–1.551   4655.7 -> 7095.5 ns
  slower  l1/prox_g_out/Float32/n2048/normal           1.514–1.528   2302.2 -> 3516.5 ns
  slower  l1/prox_g_out/Float32/n8192/normal           1.514–1.519   9424.5 -> 14271.1 ns
  slower  l1/prox_g_out/Float32/n1048576/normal        1.512–1.538   1200549.0 -> 1827906.0 ns
  slower  l1/prox_g_out/Float32/n16384/normal          1.512–1.534   18696.0 -> 28700.9 ns
  slower  l1/prox_g_out/Float64/n4096/normal           1.510–1.544   4676.8 -> 7219.0 ns
  slower  l1/prox_g_out/Float64/n1024/normal           1.504–3.778   1168.7 -> 1833.8 ns
  slower  l1/prox_g_out/Float64/n2048/normal           1.491–1.562   2334.3 -> 3570.6 ns
  slower  l1/prox_g_out/Float32/n256/normal            1.475–1.478   295.7 -> 436.7 ns
  slower  l1/prox_g_out/Float64/n1000/normal           1.462–1.510   1137.2 -> 1717.0 ns
  slower  l1/prox_g_out/Float32/n1024/normal           1.455–1.520   1161.2 -> 1760.8 ns
  slower  l1/prox_g_out/Float32/n1000/normal           1.448–1.518   1133.2 -> 1719.4 ns
  slower  l1/prox_g_out/Float64/n65536/normal          1.436–1.533   75468.9 -> 115205.9 ns
  slower  l1/prox_g_out/Float64/n32768/normal          1.426–1.545   37644.5 -> 57897.7 ns
  slower  l1/prox_g_out/Float32/n128/normal            1.411–1.422   151.9 -> 215.7 ns
  slower  enet/prox_g_out/Float32/n100000/normal       1.375–1.380   51006.9 -> 70190.9 ns
  slower  enet/prox_g_out/Float32/n8192/normal         1.375–1.381   4181.4 -> 5754.7 ns
  slower  enet/prox_g_out/Float32/n1048576/normal      1.375–1.384   535315.5 -> 737642.0 ns
  slower  enet/prox_g_out/Float32/n65536/normal        1.374–1.379   33489.9 -> 46048.9 ns
  slower  enet/prox_g_out/Float32/n262144/normal       1.374–1.382   134075.3 -> 184371.0 ns
  slower  enet/prox_g_out/Float32/n1024/normal         1.372–1.378   529.9 -> 727.6 ns
  slower  enet/prox_g_out/Float32/n16384/normal        1.371–1.390   8380.9 -> 11493.5 ns
  slower  enet/prox_g_out/Float32/n1000/normal         1.369–1.379   520.4 -> 713.6 ns
  slower  enet/prox_g_out/Float32/n256/normal          1.337–1.345   139.0 -> 186.9 ns
  slower  enet/prox_g_out/Float32/n4096/normal         1.337–1.379   2104.1 -> 2891.3 ns
  slower  enet/prox_g_out/Float32/n32768/normal        1.336–1.380   16659.2 -> 22933.4 ns
  slower  enet/prox_g_out/Float32/n2048/normal         1.333–1.382   1053.4 -> 1445.2 ns
  slower  enet/prox_out/Float32/n262144/normal         1.321–1.372   27854.1 -> 36798.6 ns
  slower  l1/prox_g_out/Float32/n64/normal             1.305–1.309   80.5 -> 105.0 ns
  slower  l1/prox_g_out/Float64/n1048576/normal        1.300–1.527   1216869.0 -> 1844868.0 ns
  slower  enet/prox_g_out/Float32/n128/normal          1.291–1.312   76.0 -> 99.2 ns
  slower  l1/prox_g_out/Float64/n16384/normal          1.267–1.541   18801.7 -> 28744.3 ns
  slower  enet/prox_g_out/Float32/n64/normal           1.264–1.281   43.5 -> 55.1 ns
  some    enet/prox_out/Float64/n65536/normal          1.030–1.389   14476.5 -> 18586.8 ns   output offsets [64, 256, 128]
  some    l1/prox_alloc/Float32/n100000/normal         0.786–1.321   16224.4 -> 16270.2 ns   output offsets [2752, 2752, 2816]
  some    enet/prox_out/Float64/n32768/normal          0.712–1.306   6859.6 -> 4908.2 ns   output offsets [640, 3328, 64]
  some    l1w/prox_in/Float64/n4/normal                0.733–1.299   8.2 -> 6.3 ns   output offsets [64, 64, 64]
  some    enet/valprox_out/Float64/n65536/normal       1.015–1.285   20471.5 -> 24825.8 ns   output offsets [64, 256, 128]
  some    enet/valprox_out/Float64/n32768/normal       0.788–1.236   9337.7 -> 7360.3 ns   output offsets [640, 3328, 64]
  some    l1/prox_g_out/Float64/n8/normal              1.022–1.187   13.1 -> 13.4 ns   output offsets [96, 96, 96]
  some    enetw/prox_out/Float64/n65536/normal         0.992–1.174   18708.3 -> 20244.0 ns   output offsets [64, 256, 128]
  some    enetw/prox_out/Float64/n32768/normal         0.995–1.163   8019.9 -> 8341.1 ns   output offsets [640, 3328, 64]
  some    l1/prox_g_out/Float64/n4/normal              1.032–1.156   8.8 -> 9.1 ns   output offsets [64, 64, 64]
  some    l1/valprox_out/Float64/n65536/normal         0.894–1.151   20022.9 -> 21112.4 ns   output offsets [64, 256, 128]
  some    enetw/prox_alloc/Float32/n2/normal           0.993–1.133   24.8 -> 26.1 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float64/n2/normal              1.042–1.125   6.6 -> 6.8 ns   output offsets [48, 48, 48]
  some    enetw/valprox_out/Float64/n4/normal          1.005–1.109   21.0 -> 21.4 ns   output offsets [64, 64, 64]
  some    l1w/prox_out/Float32/n1048576/normal         1.023–1.100   139964.9 -> 143530.4 ns   output offsets [64, 64, 128]
  some    l1/prox_g_out/Float64/n1/normal              1.050–1.100   5.5 -> 5.7 ns   output offsets [64, 64, 64]
  some    enet/prox_alloc/Float64/n4096/normal         0.977–1.099   1505.1 -> 1539.9 ns   output offsets [3840, 64, 2496]
  some    enetw/valprox_out/Float64/n65536/normal      1.007–1.093   30116.6 -> 32406.5 ns   output offsets [64, 256, 128]
  some    enetw/valprox_out/Float64/n32768/normal      0.993–1.075   13145.0 -> 13115.0 ns   output offsets [640, 3328, 64]
  some    enetw/prox_out/Float64/n1/normal             1.047–1.071   11.5 -> 12.3 ns   output offsets [64, 64, 64]
  some    l1w/prox_incopy/Float64/n1/normal            1.000–1.067   8.2 -> 8.5 ns   output offsets [64, 64, 64]
  some    l1w/prox_out/Float64/n32768/normal           0.996–1.060   8283.6 -> 8335.3 ns   output offsets [640, 3328, 64]
  some    l1w/valprox_out/Float32/n1048576/normal      1.020–1.059   231545.0 -> 236757.5 ns   output offsets [64, 64, 128]
  some    l1w/prox_incopy/Float32/n1/normal            1.029–1.059   9.3 -> 9.6 ns   output offsets [64, 64, 64]
  some    enetw/prox_g_out/Float32/n4/normal           1.018–1.057   23.4 -> 24.5 ns   output offsets [48, 48, 48]

Mi: 87 of 1000 cells slower in every process, 268 faster in every process, 31 above 1.05 in some processes only, 30 undecided (control unclean)
  slower  l1/prox_g_out/Float32/n16384/normal          1.446–1.472   18696.0 -> 27445.1 ns
  slower  l1/prox_g_out/Float32/n65536/normal          1.413–1.455   74521.6 -> 107116.6 ns
  slower  l1/prox_g_out/Float32/n32768/normal          1.410–1.499   37279.0 -> 54770.9 ns
  slower  l1/prox_g_out/Float32/n100000/normal         1.405–1.427   114099.8 -> 160888.1 ns
  slower  l1/prox_g_out/Float32/n262144/normal         1.403–1.458   299861.9 -> 424549.6 ns
  slower  l1/prox_g_out/Float32/n1048576/normal        1.396–1.476   1200549.0 -> 1708163.0 ns
  slower  enet/prox_g_out/Float32/n1048576/normal      1.382–1.383   535315.5 -> 739672.3 ns
  slower  enet/prox_g_out/Float32/n262144/normal       1.380–1.383   134075.3 -> 185085.1 ns
  slower  enet/prox_g_out/Float32/n16384/normal        1.378–1.390   8380.9 -> 11549.6 ns
  slower  enet/prox_g_out/Float32/n8192/normal         1.376–1.383   4181.4 -> 5755.2 ns
  slower  enet/prox_g_out/Float32/n100000/normal       1.375–1.382   51006.9 -> 70294.2 ns
  slower  enet/prox_g_out/Float32/n65536/normal        1.374–1.382   33489.9 -> 46111.8 ns
  slower  enet/prox_g_out/Float32/n1024/normal         1.374–1.379   529.9 -> 727.6 ns
  slower  enet/prox_g_out/Float32/n1000/normal         1.370–1.378   520.4 -> 713.9 ns
  slower  enet/prox_g_out/Float32/n32768/normal        1.344–1.381   16659.2 -> 22950.4 ns
  slower  enet/prox_g_out/Float32/n4096/normal         1.342–1.379   2104.1 -> 2896.1 ns
  slower  enet/prox_g_out/Float32/n2048/normal         1.340–1.382   1053.4 -> 1446.8 ns
  slower  enet/prox_g_out/Float32/n256/normal          1.338–1.355   139.0 -> 187.0 ns
  slower  enet/prox_g_out/Float32/n128/normal          1.300–1.329   76.0 -> 99.8 ns
  slower  l1/prox_g_out/Float32/n8192/normal           1.269–1.453   9424.5 -> 12093.4 ns
  slower  enet/prox_g_out/Float64/n1048576/normal      1.264–1.275   1193491.0 -> 1516014.0 ns
  slower  enet/prox_g_out/Float32/n64/normal           1.263–1.282   43.5 -> 55.4 ns
  slower  enet/prox_g_out/Float64/n16384/normal        1.259–1.262   18513.8 -> 23361.6 ns
  slower  enet/prox_g_out/Float64/n32768/normal        1.255–1.263   37194.1 -> 46924.1 ns
  slower  enet/prox_g_out/Float64/n100000/normal       1.255–1.263   113488.2 -> 143111.8 ns
  slower  enet/prox_g_out/Float64/n8192/normal         1.254–1.261   9280.9 -> 11696.8 ns
  slower  enet/prox_g_out/Float64/n262144/normal       1.254–1.264   297324.3 -> 374686.2 ns
  slower  enet/prox_g_out/Float64/n65536/normal        1.253–1.266   74481.4 -> 94043.4 ns
  slower  enet/prox_g_out/Float64/n2048/normal         1.253–1.261   2333.0 -> 2942.1 ns
  slower  enet/prox_g_out/Float64/n4096/normal         1.253–1.263   4653.7 -> 5864.5 ns
  slower  enet/prox_g_out/Float64/n1000/normal         1.253–1.267   1137.8 -> 1441.2 ns
  slower  enet/prox_g_out/Float64/n1024/normal         1.253–1.268   1164.9 -> 1475.6 ns
  slower  enet/prox_g_out/Float64/n256/normal          1.243–1.260   297.8 -> 375.2 ns
  slower  enet/prox_g_out/Float64/n128/normal          1.239–1.250   154.6 -> 193.2 ns
  slower  l1/prox_g_out/Float64/n2048/normal           1.231–1.247   2334.3 -> 2916.9 ns
  slower  l1/prox_g_out/Float64/n32768/normal          1.222–1.349   37644.5 -> 46033.8 ns
  slower  l1/prox_g_out/Float64/n262144/normal         1.220–1.442   301363.3 -> 369352.2 ns
  slower  l1/prox_g_out/Float64/n100000/normal         1.213–1.460   115208.3 -> 140225.7 ns
  slower  l1/prox_g_out/Float32/n4096/normal           1.212–1.412   4655.7 -> 5782.8 ns
  slower  enet/prox_g_out/Float64/n64/normal           1.210–1.235   83.0 -> 102.3 ns
  some    enet/prox_alloc/Float32/n1/normal            1.009–1.259   22.7 -> 23.0 ns   output offsets [64, 64, 64]
  some    enet/prox_alloc/Float32/n2/normal            1.011–1.248   23.3 -> 23.5 ns   output offsets [64, 64, 64]
  some    enet/prox_alloc/Float32/n4/normal            0.997–1.227   23.8 -> 24.0 ns   output offsets [48, 48, 48]
  some    l1/prox_g_out/Float64/n32/normal             1.031–1.212   39.7 -> 48.2 ns   output offsets [288, 288, 288]
  some    enet/prox_alloc/Float32/n8/normal            1.000–1.208   24.9 -> 24.9 ns   output offsets [320, 1280, 64]
  some    l1/prox_g_out/Float64/n16/normal             1.018–1.200   21.9 -> 26.3 ns   output offsets [160, 160, 160]
  some    enet/prox_alloc/Float32/n16/normal           1.004–1.199   25.7 -> 25.9 ns   output offsets [96, 192, 96]
  some    enetw/prox_alloc/Float64/n4096/normal        0.830–1.188   1528.3 -> 1529.9 ns   output offsets [3840, 64, 2496]
  some    enet/prox_alloc/Float32/n32/normal           0.990–1.179   26.0 -> 25.8 ns   output offsets [160, 160, 3360]
  some    l1/prox_g_out/Float64/n8/normal              1.021–1.167   13.1 -> 15.3 ns   output offsets [96, 96, 96]
  some    enetw/valprox_out/Float64/n65536/normal      0.982–1.155   30116.6 -> 30279.2 ns   output offsets [64, 256, 128]
  some    l1/prox_g_out/Float64/n4/normal              1.031–1.125   8.8 -> 9.8 ns   output offsets [64, 64, 64]
  some    enetw/block_loop/Float32/n128/normal         1.001–1.122   21.8 -> 22.0 ns   output offsets [544, 544, 544]
  some    enetw/prox_out/Float64/n65536/normal         0.984–1.121   18708.3 -> 18686.3 ns   output offsets [64, 256, 128]
  some    l1/prox_out/Float32/n1/normal                1.000–1.111   4.9 -> 5.5 ns   output offsets [64, 64, 64]
  some    l1w/prox_out/Float64/n2/normal               1.000–1.109   5.7 -> 6.0 ns   output offsets [48, 48, 48]
  some    enetw/prox_alloc/Float32/n64/normal          1.007–1.107   32.9 -> 33.2 ns   output offsets [288, 288, 288]
  some    l1/prox_out/Float32/n2/normal                1.000–1.100   5.5 -> 6.0 ns   output offsets [64, 64, 64]
  some    l1/prox_g_out/Float64/n2/normal              1.042–1.091   6.6 -> 7.1 ns   output offsets [48, 48, 48]
  some    enetw/prox_alloc/Float32/n2048/normal        0.986–1.087   488.5 -> 499.6 ns   output offsets [2176, 0, 256]
  some    enet/prox_alloc/Float32/n64/normal           0.992–1.085   29.5 -> 30.0 ns   output offsets [288, 288, 288]
  some    enetw/valprox_out/Float32/n64/normal         1.005–1.079   32.3 -> 32.7 ns   output offsets [288, 288, 288]
  some    enet/prox_g_out/Float32/n2/normal            1.046–1.072   18.0 -> 19.0 ns   output offsets [64, 64, 64]
  some    enetw/prox_out/Float64/n1/normal             1.023–1.071   11.5 -> 12.0 ns   output offsets [64, 64, 64]
  some    enet/prox_out/Float32/n256/normal            1.022–1.070   31.2 -> 32.5 ns   output offsets [1088, 1088, 1152]

## xeon2640v4-julia1.13.4 (6 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 5000 of 5000

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3c | A3cr | A3i | Mi | control |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 0.99 [0.97–1.02] | 0.99 [0.97–1.02] | 0.99 [0.97–1.01] | 1.00 [0.97–1.03] | 0.98 [0.96–1.00] | 1.00 |
| enet | block_loop | medium | 4 | 0.69 [0.66–0.73] | 0.70 [0.67–0.74] | 0.70 [0.67–0.74] | 0.70 [0.67–0.74] | 1.00 [1.00–1.00] | 1.00 |
| enet | prox_alloc | short | 14 | 0.99 [0.92–1.02] | 0.99 [0.93–1.03] | 0.99 [0.92–1.04] | 0.99 [0.92–1.03] | 1.02 [1.00–1.06] | 1.00 |
| enet | prox_alloc | medium | 14 | 0.97 [0.93–0.99] | 0.97 [0.92–1.01] | 0.98 [0.92–1.01] | 0.97 [0.93–1.00] | 1.01 [1.00–1.04] | 1.00 |
| enet | prox_alloc | large | 12 | 1.00 [0.98–1.01] | 1.00 [0.99–1.02] | 1.00 [0.98–1.01] | 1.00 [0.98–1.04] | 1.00 [0.99–1.01] | 1.00 |
| enet | prox_g_out | short | 14 | 1.00 [0.99–1.02] | 1.00 [0.99–1.04] | 1.00 [0.99–1.02] | 1.06 [1.01–1.13] | 1.06 [1.01–1.13] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.02 [1.00–1.04] | 1.02 [1.00–1.04] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.03 [1.00–1.08] | 1.03 [1.00–1.09] | 1.00 |
| enet | prox_in | short | 14 | 0.86 [0.69–1.02] | 0.86 [0.69–1.04] | 0.87 [0.68–1.03] | 0.59 [0.17–1.06] | 0.62 [0.19–1.04] | 1.00 |
| enet | prox_in | medium | 14 | 0.67 [0.67–0.69] | 0.71 [0.67–1.00] | 0.67 [0.67–0.69] | 0.13 [0.08–0.20] | 0.19 [0.13–0.28] | 1.00 |
| enet | prox_in | large | 12 | 0.67 [0.67–0.67] | 1.00 [1.00–1.00] | 0.67 [0.67–0.67] | 0.13 [0.08–0.21] | 0.18 [0.13–0.26] | 1.00 |
| enet | prox_incopy | short | 14 | 0.87 [0.71–1.01] | 0.88 [0.71–1.04] | 0.88 [0.72–1.03] | 0.64 [0.22–1.04] | 0.66 [0.24–1.04] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.69 [0.68–0.73] | 0.73 [0.68–1.00] | 0.69 [0.68–0.73] | 0.18 [0.11–0.31] | 0.23 [0.15–0.38] | 1.00 |
| enet | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 1.00 [1.00–1.00] | 0.72 [0.69–0.75] | 0.27 [0.16–0.56] | 0.31 [0.20–0.56] | 1.00 |
| enet | prox_out | short | 14 | 0.96 [0.74–1.02] | 0.97 [0.75–1.04] | 0.98 [0.76–1.04] | 0.99 [0.75–1.06] | 1.03 [1.01–1.05] | 1.00 |
| enet | prox_out | medium | 14 | 0.69 [0.66–0.74] | 0.73 [0.67–1.00] | 0.69 [0.67–0.75] | 0.69 [0.67–0.76] | 1.00 [0.98–1.02] | 1.00 |
| enet | prox_out | large | 12 | 0.89 [0.68–0.99] | 1.00 [1.00–1.00] | 0.89 [0.67–0.99] | 0.89 [0.68–0.99] | 1.00 [1.00–1.00] | 1.00 |
| enet | sparse_prox | large | 2 | 0.94 [0.93–0.94] | 0.94 [0.94–0.94] | 0.93 [0.93–0.93] | 0.94 [0.93–0.94] | 1.00 [1.00–1.00] | 1.00 |
| enet | valprox_out | short | 14 | 0.97 [0.82–1.01] | 0.99 [0.84–1.04] | 0.99 [0.85–1.04] | 0.99 [0.84–1.04] | 1.02 [1.00–1.03] | 1.00 |
| enet | valprox_out | medium | 14 | 0.76 [0.74–0.82] | 0.79 [0.73–1.00] | 0.76 [0.74–0.82] | 0.76 [0.74–0.84] | 1.00 [0.98–1.02] | 1.00 |
| enet | valprox_out | large | 12 | 0.91 [0.75–0.99] | 1.00 [1.00–1.00] | 0.91 [0.75–0.99] | 0.91 [0.75–0.99] | 1.00 [1.00–1.00] | 1.00 |
| enetw | block_loop | short | 4 | 1.01 [0.90–1.08] | 1.03 [0.91–1.08] | 1.01 [0.90–1.08] | 1.03 [0.91–1.08] | 1.00 [0.98–1.02] | 1.00 |
| enetw | block_loop | medium | 4 | 0.86 [0.79–0.97] | 0.85 [0.78–0.97] | 0.86 [0.79–0.97] | 0.85 [0.78–0.98] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_alloc | short | 14 | 0.97 [0.92–1.02] | 0.97 [0.92–1.01] | 0.98 [0.92–1.01] | 0.99 [0.93–1.06] | 1.01 [0.99–1.03] | 0.99 |
| enetw | prox_alloc | medium | 14 | 0.97 [0.91–1.00] | 0.97 [0.91–1.01] | 0.97 [0.91–1.01] | 0.97 [0.92–0.99] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_alloc | large | 12 | 1.00 [1.00–1.01] | 1.00 [0.99–1.01] | 1.00 [0.99–1.03] | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_g_out | short | 14 | 1.00 [1.00–1.01] | 1.00 [0.98–1.01] | 1.00 [0.98–1.01] | 1.02 [1.00–1.04] | 1.03 [1.00–1.06] | 1.00 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.03 [1.01–1.05] | 1.03 [1.01–1.05] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.03 [1.01–1.07] | 1.03 [1.01–1.07] | 1.00 |
| enetw | prox_in | short | 14 | 0.86 [0.71–1.00] | 0.86 [0.71–1.00] | 0.88 [0.71–1.03] | 0.60 [0.20–1.05] | 0.63 [0.23–1.03] | 1.00 |
| enetw | prox_in | medium | 14 | 0.68 [0.67–0.70] | 0.72 [0.67–1.00] | 0.68 [0.67–0.70] | 0.14 [0.09–0.22] | 0.19 [0.13–0.29] | 1.00 |
| enetw | prox_in | large | 12 | 0.67 [0.67–0.68] | 1.00 [1.00–1.00] | 0.67 [0.67–0.68] | 0.20 [0.09–0.31] | 0.22 [0.13–0.32] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.87 [0.73–1.00] | 0.87 [0.73–1.00] | 0.88 [0.73–1.02] | 0.65 [0.25–1.03] | 0.66 [0.28–1.02] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.70 [0.68–0.72] | 0.73 [0.68–1.00] | 0.70 [0.68–0.72] | 0.19 [0.11–0.30] | 0.24 [0.15–0.36] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 1.00 [1.00–1.00] | 0.72 [0.69–0.75] | 0.32 [0.16–0.61] | 0.33 [0.19–0.61] | 1.00 |
| enetw | prox_out | short | 14 | 0.95 [0.80–1.01] | 0.95 [0.80–1.01] | 0.97 [0.81–1.03] | 0.97 [0.81–1.03] | 1.02 [1.00–1.03] | 1.00 |
| enetw | prox_out | medium | 14 | 0.74 [0.67–0.81] | 0.77 [0.67–1.00] | 0.74 [0.67–0.82] | 0.75 [0.68–0.82] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_out | large | 12 | 0.97 [0.81–1.00] | 1.00 [1.00–1.00] | 0.97 [0.81–1.00] | 0.97 [0.81–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | sparse_prox | large | 2 | 0.94 [0.93–0.96] | 0.94 [0.93–0.96] | 0.94 [0.93–0.96] | 0.94 [0.93–0.96] | 1.00 [1.00–1.00] | 1.00 |
| enetw | valprox_out | short | 14 | 0.96 [0.86–1.00] | 0.97 [0.86–1.00] | 0.98 [0.86–1.02] | 0.99 [0.86–1.03] | 1.01 [0.99–1.05] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.81 [0.78–0.86] | 0.83 [0.78–1.00] | 0.81 [0.78–0.86] | 0.82 [0.78–0.88] | 1.00 [1.00–1.03] | 1.00 |
| enetw | valprox_out | large | 12 | 0.98 [0.86–1.00] | 1.00 [1.00–1.00] | 0.98 [0.86–1.00] | 0.98 [0.86–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 0.98 [0.92–1.04] | 0.98 [0.92–1.04] | 0.98 [0.92–1.04] | 0.98 [0.92–1.04] | 1.00 [0.98–1.00] | 1.00 |
| l1 | block_loop | medium | 4 | 0.66 [0.65–0.68] | 0.66 [0.65–0.67] | 0.66 [0.65–0.68] | 0.66 [0.65–0.68] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.98 [0.93–1.03] | 1.00 [0.93–1.04] | 1.00 [0.93–1.06] | 0.99 [0.93–1.04] | 1.01 [1.00–1.03] | 1.00 |
| l1 | prox_alloc | medium | 14 | 0.98 [0.93–1.01] | 0.98 [0.93–1.00] | 0.98 [0.93–1.00] | 0.97 [0.93–0.99] | 1.00 [0.98–1.01] | 1.00 |
| l1 | prox_alloc | large | 12 | 1.00 [0.98–1.02] | 1.00 [0.99–1.01] | 1.00 [0.98–1.03] | 1.00 [0.98–1.02] | 1.00 [0.99–1.02] | 1.00 |
| l1 | prox_g_out | short | 14 | 0.90 [0.80–1.04] | 0.90 [0.80–1.04] | 0.91 [0.80–1.04] | 0.91 [0.80–1.06] | 1.03 [1.01–1.06] | 1.00 |
| l1 | prox_g_out | medium | 14 | 0.78 [0.78–0.79] | 0.78 [0.78–0.79] | 0.78 [0.78–0.79] | 0.79 [0.79–0.81] | 1.00 [1.00–1.01] | 1.00 |
| l1 | prox_g_out | large | 12 | 0.78 [0.78–0.80] | 0.78 [0.78–0.80] | 0.78 [0.78–0.80] | 0.80 [0.79–0.85] | 1.01 [1.00–1.03] | 1.00 |
| l1 | prox_in | short | 14 | 0.83 [0.67–1.13] | 0.85 [0.67–1.17] | 0.86 [0.67–1.19] | 0.48 [0.11–1.12] | 0.52 [0.14–1.10] | 1.00 |
| l1 | prox_in | medium | 14 | 0.67 [0.67–0.67] | 0.71 [0.66–1.00] | 0.67 [0.66–0.67] | 0.12 [0.08–0.17] | 0.18 [0.13–0.25] | 1.00 |
| l1 | prox_in | large | 12 | 0.67 [0.67–0.67] | 1.00 [1.00–1.00] | 0.67 [0.67–0.67] | 0.13 [0.08–0.21] | 0.18 [0.13–0.25] | 1.00 |
| l1 | prox_incopy | short | 14 | 0.84 [0.69–1.05] | 0.86 [0.69–1.12] | 0.89 [0.70–1.20] | 0.55 [0.16–1.09] | 0.58 [0.19–1.07] | 1.00 |
| l1 | prox_incopy | medium | 14 | 0.69 [0.68–0.73] | 0.72 [0.68–1.00] | 0.69 [0.68–0.73] | 0.17 [0.11–0.31] | 0.23 [0.15–0.38] | 1.00 |
| l1 | prox_incopy | large | 12 | 0.72 [0.69–0.75] | 1.00 [1.00–1.00] | 0.72 [0.69–0.75] | 0.27 [0.16–0.56] | 0.31 [0.20–0.56] | 1.00 |
| l1 | prox_out | short | 14 | 0.96 [0.65–1.15] | 0.97 [0.65–1.17] | 0.97 [0.65–1.17] | 0.98 [0.65–1.17] | 1.05 [1.01–1.09] | 1.01 |
| l1 | prox_out | medium | 14 | 0.67 [0.65–0.67] | 0.71 [0.65–1.00] | 0.67 [0.65–0.67] | 0.67 [0.65–0.67] | 1.00 [1.00–1.01] | 1.00 |
| l1 | prox_out | large | 12 | 0.88 [0.67–0.99] | 1.00 [1.00–1.00] | 0.88 [0.67–0.99] | 0.89 [0.67–0.99] | 1.00 [1.00–1.00] | 1.00 |
| l1 | sparse_prox | large | 2 | 0.95 [0.95–0.96] | 0.95 [0.95–0.96] | 0.96 [0.95–0.96] | 0.95 [0.95–0.96] | 1.00 [1.00–1.00] | 1.00 |
| l1 | valprox_out | short | 14 | 0.95 [0.75–1.04] | 0.95 [0.76–1.04] | 0.97 [0.76–1.09] | 0.97 [0.75–1.07] | 1.01 [1.00–1.04] | 1.00 |
| l1 | valprox_out | medium | 14 | 0.74 [0.73–0.76] | 0.77 [0.72–1.00] | 0.74 [0.72–0.76] | 0.74 [0.73–0.76] | 1.00 [1.00–1.00] | 1.00 |
| l1 | valprox_out | large | 12 | 0.91 [0.75–0.99] | 1.00 [1.00–1.00] | 0.91 [0.75–0.99] | 0.91 [0.75–0.99] | 1.00 [1.00–1.00] | 1.00 |
| l1w | block_loop | short | 4 | 0.98 [0.90–1.03] | 0.97 [0.90–1.03] | 0.98 [0.90–1.03] | 0.98 [0.90–1.03] | 1.00 [1.00–1.02] | 1.00 |
| l1w | block_loop | medium | 4 | 0.68 [0.67–0.69] | 0.68 [0.67–0.69] | 0.68 [0.67–0.69] | 0.68 [0.67–0.69] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 0.98 [0.92–1.01] | 0.98 [0.93–1.02] | 0.98 [0.92–1.02] | 0.98 [0.92–1.03] | 1.01 [0.99–1.03] | 1.00 |
| l1w | prox_alloc | medium | 14 | 0.97 [0.93–1.00] | 0.97 [0.93–1.00] | 0.98 [0.93–1.00] | 0.97 [0.93–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | large | 12 | 1.00 [0.99–1.00] | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 0.99 [0.98–1.01] | 1.00 [0.98–1.01] | 1.00 |
| l1w | prox_g_out | short | 14 | 0.91 [0.85–1.02] | 0.91 [0.84–1.02] | 0.91 [0.84–1.02] | 0.92 [0.83–1.05] | 1.01 [0.97–1.04] | 1.00 |
| l1w | prox_g_out | medium | 14 | 0.84 [0.83–0.84] | 0.84 [0.83–0.84] | 0.84 [0.83–0.84] | 0.84 [0.83–0.84] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_g_out | large | 12 | 0.84 [0.83–0.86] | 0.84 [0.83–0.86] | 0.84 [0.83–0.86] | 0.84 [0.83–0.86] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_in | short | 14 | 0.82 [0.68–1.04] | 0.83 [0.65–1.06] | 0.83 [0.69–1.08] | 0.46 [0.11–1.06] | 0.50 [0.15–1.05] | 1.00 |
| l1w | prox_in | medium | 14 | 0.67 [0.67–0.68] | 0.71 [0.67–1.00] | 0.67 [0.67–0.68] | 0.12 [0.09–0.18] | 0.18 [0.13–0.26] | 1.00 |
| l1w | prox_in | large | 12 | 0.67 [0.67–0.67] | 1.00 [1.00–1.00] | 0.67 [0.67–0.67] | 0.20 [0.09–0.31] | 0.22 [0.13–0.31] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.84 [0.70–1.04] | 0.86 [0.70–1.10] | 0.87 [0.70–1.14] | 0.56 [0.18–1.05] | 0.59 [0.21–1.05] | 1.00 |
| l1w | prox_incopy | medium | 14 | 0.69 [0.68–0.71] | 0.73 [0.68–1.00] | 0.69 [0.68–0.71] | 0.18 [0.11–0.30] | 0.23 [0.15–0.36] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.72 [0.69–0.80] | 1.00 [1.00–1.00] | 0.72 [0.69–0.80] | 0.32 [0.16–0.64] | 0.33 [0.19–0.63] | 1.00 |
| l1w | prox_out | short | 14 | 0.92 [0.69–1.04] | 0.95 [0.70–1.08] | 0.95 [0.70–1.08] | 0.93 [0.70–1.04] | 1.00 [0.97–1.04] | 1.00 |
| l1w | prox_out | medium | 14 | 0.72 [0.67–0.81] | 0.74 [0.67–1.00] | 0.72 [0.67–0.81] | 0.72 [0.67–0.81] | 1.00 [1.00–1.01] | 1.00 |
| l1w | prox_out | large | 12 | 0.97 [0.80–1.00] | 1.00 [1.00–1.00] | 0.97 [0.80–1.00] | 0.97 [0.80–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | sparse_prox | large | 2 | 0.95 [0.94–0.96] | 0.95 [0.94–0.96] | 0.95 [0.94–0.96] | 0.95 [0.94–0.96] | 1.00 [1.00–1.00] | 1.00 |
| l1w | valprox_out | short | 14 | 0.95 [0.85–1.03] | 0.97 [0.84–1.05] | 0.98 [0.84–1.07] | 0.97 [0.86–1.02] | 1.02 [0.99–1.06] | 0.99 |
| l1w | valprox_out | medium | 14 | 0.83 [0.81–0.88] | 0.85 [0.80–1.00] | 0.83 [0.81–0.88] | 0.84 [0.81–0.88] | 1.00 [1.00–1.01] | 1.00 |
| l1w | valprox_out | large | 12 | 0.98 [0.87–1.00] | 1.00 [1.00–1.00] | 0.98 [0.87–1.00] | 0.98 [0.87–1.00] | 1.00 [1.00–1.00] | 1.00 |

A3M4: 3 of 1000 cells slower in every process, 480 faster in every process, 7 above 1.05 in some processes only, 59 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.125–1.125   7.1 -> 8.0 ns
  slower  enetw/block_loop/Float64/n16/normal          1.074–1.083   18.0 -> 19.4 ns
  slower  enetw/block_loop/Float32/n4/normal           1.059–1.092   14.7 -> 15.9 ns
  some    enetw/block_loop/Float64/n4/normal           1.000–1.082   14.7 -> 14.7 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    l1/prox_incopy/Float64/n1/normal             1.035–1.079   11.7 -> 12.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.034–1.069   8.6 -> 9.0 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1w/prox_alloc/Float64/n16384/normal         0.993–1.069   15336.8 -> 15237.4 ns   output offsets [64, 3008, 2432, 3776, 512, 2688]
  some    l1w/valprox_out/Float32/n4/normal            1.000–1.062   19.3 -> 19.5 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_out/Float64/n1/normal              1.017–1.053   17.0 -> 17.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_in/Float64/n1/normal               1.017–1.053   17.0 -> 17.4 ns   output offsets [64, 64, 64, 64, 64, 64]

A3c: 5 of 1000 cells slower in every process, 324 faster in every process, 12 above 1.05 in some processes only, 59 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.125–1.167   7.1 -> 8.1 ns
  slower  l1/prox_incopy/Float32/n1/normal             1.114–1.159   13.0 -> 14.6 ns
  slower  enetw/block_loop/Float64/n16/normal          1.066–1.100   18.0 -> 19.2 ns
  slower  l1/prox_incopy/Float64/n1/normal             1.059–1.105   11.7 -> 12.4 ns
  slower  enetw/block_loop/Float32/n4/normal           1.059–1.100   14.7 -> 16.1 ns
  some    enetw/block_loop/Float64/n4/normal           1.020–1.785   14.7 -> 15.6 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    l1w/prox_incopy/Float32/n2/normal            1.039–1.135   14.7 -> 15.6 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.035–1.103   8.6 -> 9.1 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_incopy/Float32/n2/normal             1.040–1.078   14.1 -> 15.0 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enet/prox_in/Float64/n1/normal               1.017–1.070   17.0 -> 17.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float64/n1/normal              1.017–1.070   17.0 -> 17.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/valprox_out/Float32/n4/normal            1.045–1.062   19.3 -> 20.4 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_alloc/Float32/n64/normal           0.925–1.059   56.4 -> 53.7 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    enet/prox_out/Float32/n1/normal              1.000–1.059   15.3 -> 15.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float64/n8/normal              0.985–1.055   19.8 -> 19.8 ns   output offsets [96, 288, 96, 96, 96, 96]
  some    enet/prox_out/Float32/n2/normal              1.000–1.055   16.5 -> 16.7 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n8/normal              0.999–1.052   29.5 -> 30.0 ns   output offsets [96, 288, 96, 96, 96, 96]

A3cr: 8 of 1000 cells slower in every process, 472 faster in every process, 13 above 1.05 in some processes only, 60 undecided (control unclean)
  slower  l1/prox_incopy/Float32/n1/normal             1.204–1.227   13.0 -> 15.6 ns
  slower  l1/prox_out/Float32/n1/normal                1.125–1.167   7.1 -> 8.0 ns
  slower  l1/prox_incopy/Float32/n2/normal             1.089–1.177   14.1 -> 16.0 ns
  slower  l1/prox_incopy/Float64/n1/normal             1.083–1.132   11.7 -> 12.7 ns
  slower  l1w/prox_incopy/Float32/n2/normal            1.078–1.114   14.7 -> 15.9 ns
  slower  enetw/block_loop/Float64/n16/normal          1.073–1.083   18.0 -> 19.4 ns
  slower  l1/prox_out/Float32/n4/normal                1.069–1.103   8.6 -> 9.1 ns
  slower  enetw/block_loop/Float32/n4/normal           1.059–1.100   14.7 -> 15.9 ns
  some    l1/prox_incopy/Float32/n4/normal             1.029–1.113   17.2 -> 18.6 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enetw/block_loop/Float64/n4/normal           1.000–1.082   14.7 -> 14.7 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    l1w/valprox_out/Float32/n4/normal            1.030–1.077   19.3 -> 20.1 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_in/Float64/n1/normal               1.017–1.076   17.0 -> 17.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n2/normal            0.966–1.074   34.5 -> 34.4 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enet/prox_out/Float64/n1/normal              1.017–1.070   17.0 -> 17.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_in/Float32/n1/normal               1.000–1.058   15.3 -> 15.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float32/n1/normal              1.000–1.058   15.3 -> 15.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_incopy/Float32/n1/normal          0.983–1.056   24.8 -> 24.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n8/normal              1.011–1.056   29.5 -> 30.4 ns   output offsets [96, 288, 96, 96, 96, 96]
  some    enetw/prox_alloc/Float64/n8192/normal        0.980–1.055   6560.7 -> 6451.8 ns   output offsets [384, 64, 2624, 4032, 1920, 64]
  some    enet/prox_in/Float32/n2/normal               0.982–1.054   16.5 -> 17.0 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enet/prox_out/Float32/n2/normal              0.982–1.054   16.5 -> 16.8 ns   output offsets [64, 96, 64, 64, 64, 64]

A3i: 11 of 1000 cells slower in every process, 481 faster in every process, 18 above 1.05 in some processes only, 71 undecided (control unclean)
  slower  l1/prox_out/Float32/n1/normal                1.167–1.167   7.1 -> 8.3 ns
  slower  enet/prox_g_out/Float32/n4/normal            1.122–1.154   31.2 -> 35.4 ns
  slower  l1/prox_incopy/Float32/n1/normal             1.091–1.091   13.0 -> 14.2 ns
  slower  enet/prox_g_out/Float32/n262144/normal       1.084–1.085   309773.8 -> 336038.2 ns
  slower  enet/prox_g_out/Float32/n1048576/normal      1.083–1.086   1237713.5 -> 1342300.5 ns
  slower  enetw/block_loop/Float32/n4/normal           1.078–1.100   14.7 -> 16.2 ns
  slower  l1/prox_out/Float32/n4/normal                1.069–1.069   8.6 -> 9.1 ns
  slower  enetw/prox_g_out/Float32/n262144/normal      1.067–1.068   319058.0 -> 340453.2 ns
  slower  enetw/prox_g_out/Float32/n1048576/normal     1.067–1.068   1275288.5 -> 1361020.5 ns
  slower  enetw/block_loop/Float64/n16/normal          1.066–1.092   18.0 -> 19.2 ns
  slower  enet/prox_in/Float32/n1/normal               1.058–1.077   15.3 -> 16.2 ns
  some    enetw/prox_g_out/Float64/n8192/normal        1.008–1.187   26799.0 -> 27075.5 ns   output offsets [384, 64, 2624, 4032, 1920, 64]
  some    enet/prox_g_out/Float64/n8192/normal         1.004–1.134   26859.4 -> 26981.8 ns   output offsets [384, 64, 2624, 4032, 1920, 64]
  some    enet/prox_in/Float64/n1/normal               1.017–1.088   17.0 -> 17.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_incopy/Float64/n1/normal             1.038–1.079   11.7 -> 12.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/valprox_out/Float32/n8/normal          0.982–1.074   32.7 -> 33.5 ns   output offsets [64, 64, 64, 64, 128, 640]
  some    l1w/prox_g_out/Float32/n1/normal             1.036–1.074   8.3 -> 8.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float64/n1/normal              1.000–1.070   17.0 -> 17.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_g_out/Float32/n32768/normal        1.034–1.065   38544.1 -> 39910.8 ns   output offsets [3520, 640, 192, 768, 704, 0]
  some    enet/prox_out/Float64/n4/normal              1.016–1.064   18.6 -> 19.3 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    l1w/prox_g_out/Float32/n2/normal             1.012–1.061   10.2 -> 10.3 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enetw/block_loop/Float64/n4/normal           1.000–1.060   14.7 -> 15.6 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    enet/prox_g_out/Float32/n256/normal          1.031–1.059   313.0 -> 324.1 ns   output offsets [3264, 1152, 1088, 1088, 1088, 1088]
  some    enetw/prox_alloc/Float64/n2/normal           0.990–1.059   38.6 -> 39.2 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_out/Float32/n1/normal              1.019–1.059   15.3 -> 16.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float64/n8/normal              1.000–1.055   19.8 -> 20.1 ns   output offsets [96, 288, 96, 96, 96, 96]
  some    enet/prox_out/Float32/n2/normal              1.018–1.055   16.5 -> 17.3 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enet/prox_in/Float32/n2/normal               1.000–1.054   16.5 -> 16.8 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enetw/prox_g_out/Float32/n256/normal         1.044–1.053   318.6 -> 333.1 ns   output offsets [3264, 1152, 1088, 1088, 1088, 1088]

Mi: 6 of 1000 cells slower in every process, 253 faster in every process, 27 above 1.05 in some processes only, 38 undecided (control unclean)
  slower  enet/prox_g_out/Float32/n4/normal            1.122–1.160   31.2 -> 35.4 ns
  slower  enet/prox_g_out/Float32/n262144/normal       1.084–1.086   309773.8 -> 336066.8 ns
  slower  enet/prox_g_out/Float32/n1048576/normal      1.084–1.086   1237713.5 -> 1342906.5 ns
  slower  l1/prox_incopy/Float32/n1/normal             1.068–1.068   13.0 -> 13.9 ns
  slower  enetw/prox_g_out/Float32/n262144/normal      1.067–1.068   319058.0 -> 340501.0 ns
  slower  enetw/prox_g_out/Float32/n1048576/normal     1.067–1.068   1275288.5 -> 1361359.0 ns
  some    enet/prox_alloc/Float32/n2/normal            0.972–1.252   34.5 -> 36.4 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enetw/prox_g_out/Float64/n16/normal          1.021–1.196   67.9 -> 71.0 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    enetw/prox_g_out/Float64/n8192/normal        1.008–1.185   26799.0 -> 27073.8 ns   output offsets [384, 64, 2624, 4032, 1920, 64]
  some    enetw/prox_g_out/Float64/n8/normal           1.023–1.146   42.6 -> 44.1 ns   output offsets [96, 288, 96, 96, 96, 96]
  some    enetw/valprox_out/Float64/n128/normal        0.993–1.138   105.2 -> 105.4 ns   output offsets [1088, 1088, 1664, 1088, 2176, 256]
  some    enet/prox_g_out/Float64/n8192/normal         1.004–1.131   26859.4 -> 26982.1 ns   output offsets [384, 64, 2624, 4032, 1920, 64]
  some    l1w/prox_alloc/Float64/n16384/normal         0.993–1.104   15336.8 -> 16192.9 ns   output offsets [64, 3008, 2432, 3776, 512, 2688]
  some    enetw/valprox_out/Float32/n8/normal          0.991–1.087   32.7 -> 33.8 ns   output offsets [64, 64, 64, 64, 128, 640]
  some    l1/prox_out/Float32/n1/normal                1.042–1.084   7.1 -> 7.5 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_incopy/Float64/n1/normal             1.038–1.079   11.7 -> 12.1 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_g_out/Float64/n4/normal              1.036–1.078   13.8 -> 14.6 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    l1/prox_alloc/Float64/n2/normal              1.013–1.076   25.9 -> 26.5 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enetw/valprox_out/Float32/n64/normal         1.042–1.072   44.2 -> 46.3 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    enet/prox_in/Float64/n1/normal               1.017–1.070   17.0 -> 17.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_out/Float32/n4/normal                1.034–1.069   8.6 -> 9.0 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_g_out/Float32/n32768/normal        1.034–1.065   38544.1 -> 39925.7 ns   output offsets [3520, 640, 192, 768, 704, 0]
  some    enet/prox_out/Float64/n4/normal              1.016–1.065   18.6 -> 19.3 ns   output offsets [64, 128, 64, 64, 64, 64]
  some    l1w/prox_out/Float32/n256/normal             1.004–1.064   60.8 -> 61.1 ns   output offsets [3264, 1152, 1088, 1088, 1088, 1088]
  some    enetw/prox_alloc/Float64/n8192/normal        0.995–1.063   6560.7 -> 6604.1 ns   output offsets [384, 64, 2624, 4032, 1920, 64]
  some    enet/prox_out/Float64/n8/normal              1.015–1.061   19.8 -> 20.5 ns   output offsets [96, 288, 96, 96, 96, 96]
  some    enet/prox_alloc/Float32/n64/normal           1.010–1.056   56.4 -> 57.7 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    l1/prox_alloc/Float64/n8/normal              1.004–1.056   29.5 -> 30.3 ns   output offsets [96, 288, 96, 96, 96, 96]
  some    enet/prox_g_out/Float32/n256/normal          1.033–1.056   313.0 -> 325.6 ns   output offsets [3264, 1152, 1088, 1088, 1088, 1088]
  some    enet/prox_out/Float32/n2/normal              1.035–1.055   16.5 -> 17.4 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enetw/prox_g_out/Float32/n256/normal         1.044–1.054   318.6 -> 333.8 ns   output offsets [3264, 1152, 1088, 1088, 1088, 1088]

## xeon4116-julia1.13.4 (6 processes used; reference M4; invalid (case, implementation) pairs: 0)

cells with complete, valid data: 5000 of 5000

Geometric means of tree/reference (cases), both element types:

| family | operation | size | cases | A3M4 | A3c | A3cr | A3i | Mi | control |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| enet | block_loop | short | 4 | 1.00 [0.95–1.04] | 1.00 [0.95–1.03] | 1.00 [0.95–1.04] | 0.99 [0.95–1.03] | 0.99 [0.93–1.02] | 0.99 |
| enet | block_loop | medium | 4 | 0.85 [0.82–0.91] | 0.84 [0.82–0.88] | 0.85 [0.82–0.92] | 0.84 [0.82–0.88] | 1.00 [0.99–1.00] | 1.00 |
| enet | prox_alloc | short | 14 | 0.99 [0.97–1.01] | 0.99 [0.97–1.00] | 1.00 [0.97–1.02] | 1.00 [0.98–1.01] | 1.01 [1.00–1.02] | 1.00 |
| enet | prox_alloc | medium | 14 | 1.01 [0.97–1.03] | 1.00 [0.93–1.03] | 1.01 [0.96–1.05] | 1.01 [0.99–1.05] | 1.00 [0.97–1.01] | 1.00 |
| enet | prox_alloc | large | 12 | 1.04 [1.01–1.06] | 1.00 [0.99–1.01] | 1.03 [1.02–1.06] | 1.03 [1.01–1.07] | 1.00 [0.99–1.00] | 1.01 |
| enet | prox_g_out | short | 14 | 0.99 [0.94–1.05] | 1.00 [0.98–1.05] | 1.00 [0.91–1.05] | 1.04 [1.00–1.16] | 1.05 [1.00–1.17] | 1.00 |
| enet | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.01 [1.00–1.03] | 1.01 [1.00–1.03] | 1.00 |
| enet | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.01 [1.00–1.06] | 1.01 [1.00–1.05] | 1.00 |
| enet | prox_in | short | 14 | 0.92 [0.82–1.05] | 0.91 [0.74–1.05] | 0.96 [0.84–1.07] | 0.66 [0.25–1.02] | 0.68 [0.26–1.03] | 1.00 |
| enet | prox_in | medium | 14 | 0.77 [0.76–0.81] | 0.80 [0.76–1.00] | 0.77 [0.76–0.80] | 0.16 [0.10–0.28] | 0.19 [0.12–0.33] | 1.00 |
| enet | prox_in | large | 12 | 0.77 [0.76–0.77] | 1.00 [0.99–1.00] | 0.77 [0.76–0.78] | 0.18 [0.11–0.34] | 0.22 [0.14–0.37] | 0.98 |
| enet | prox_incopy | short | 14 | 0.92 [0.81–1.02] | 0.93 [0.82–1.06] | 0.95 [0.82–1.06] | 0.70 [0.31–1.02] | 0.72 [0.33–1.03] | 1.00 |
| enet | prox_incopy | medium | 14 | 0.79 [0.77–0.82] | 0.82 [0.77–1.00] | 0.79 [0.77–0.81] | 0.22 [0.12–0.35] | 0.25 [0.14–0.40] | 1.00 |
| enet | prox_incopy | large | 12 | 0.81 [0.78–0.92] | 1.00 [1.00–1.00] | 0.82 [0.78–0.92] | 0.32 [0.18–0.78] | 0.35 [0.21–0.79] | 0.99 |
| enet | prox_out | short | 14 | 0.99 [0.89–1.04] | 0.98 [0.87–1.05] | 1.02 [0.89–1.09] | 0.97 [0.88–1.02] | 1.00 [0.95–1.04] | 1.01 |
| enet | prox_out | medium | 14 | 0.84 [0.80–0.90] | 0.85 [0.78–1.01] | 0.84 [0.80–0.90] | 0.83 [0.80–0.88] | 1.01 [0.99–1.05] | 1.01 |
| enet | prox_out | large | 12 | 0.89 [0.82–1.02] | 1.00 [1.00–1.00] | 0.89 [0.82–1.02] | 0.89 [0.81–1.02] | 1.01 [1.00–1.01] | 1.00 |
| enet | sparse_prox | large | 2 | 0.88 [0.85–0.92] | 0.89 [0.84–0.93] | 0.89 [0.85–0.93] | 0.88 [0.85–0.91] | 1.01 [0.98–1.05] | 1.01 |
| enet | valprox_out | short | 14 | 0.99 [0.92–1.02] | 0.98 [0.91–1.02] | 1.02 [0.92–1.08] | 0.99 [0.92–1.02] | 1.00 [0.97–1.04] | 1.00 |
| enet | valprox_out | medium | 14 | 0.88 [0.85–0.91] | 0.90 [0.85–1.00] | 0.88 [0.85–0.91] | 0.88 [0.85–0.91] | 1.00 [0.99–1.01] | 1.00 |
| enet | valprox_out | large | 12 | 0.92 [0.86–1.02] | 1.00 [1.00–1.00] | 0.92 [0.86–1.02] | 0.92 [0.86–1.02] | 1.00 [1.00–1.00] | 1.00 |
| enetw | block_loop | short | 4 | 0.99 [0.93–1.05] | 1.03 [0.97–1.12] | 1.00 [0.93–1.05] | 1.05 [0.98–1.13] | 1.00 [0.98–1.03] | 1.00 |
| enetw | block_loop | medium | 4 | 0.92 [0.89–0.95] | 0.92 [0.89–0.95] | 0.91 [0.90–0.94] | 0.91 [0.89–0.94] | 1.00 [1.00–1.00] | 1.00 |
| enetw | prox_alloc | short | 14 | 1.00 [0.98–1.01] | 0.99 [0.97–1.00] | 1.00 [0.98–1.01] | 1.01 [0.99–1.03] | 1.01 [1.00–1.03] | 1.00 |
| enetw | prox_alloc | medium | 14 | 1.00 [0.97–1.01] | 1.00 [0.98–1.00] | 1.00 [0.98–1.01] | 1.00 [0.99–1.02] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_alloc | large | 12 | 1.01 [1.00–1.02] | 1.00 [0.97–1.01] | 1.01 [1.00–1.02] | 1.01 [1.00–1.03] | 1.00 [0.99–1.01] | 1.00 |
| enetw | prox_g_out | short | 14 | 1.01 [0.99–1.03] | 1.01 [0.98–1.03] | 1.00 [0.98–1.02] | 1.05 [1.00–1.14] | 1.06 [1.01–1.13] | 0.99 |
| enetw | prox_g_out | medium | 14 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.02 [1.00–1.06] | 1.02 [1.00–1.06] | 1.00 |
| enetw | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.00 [1.00–1.00] | 1.02 [1.00–1.05] | 1.02 [1.00–1.05] | 1.00 |
| enetw | prox_in | short | 14 | 0.95 [0.87–1.02] | 0.92 [0.87–0.97] | 0.97 [0.87–1.07] | 0.64 [0.24–1.01] | 0.68 [0.26–1.05] | 1.00 |
| enetw | prox_in | medium | 14 | 0.84 [0.84–0.86] | 0.86 [0.83–1.00] | 0.84 [0.84–0.86] | 0.17 [0.11–0.28] | 0.19 [0.12–0.30] | 1.00 |
| enetw | prox_in | large | 12 | 0.84 [0.83–0.86] | 1.00 [1.00–1.00] | 0.85 [0.84–0.87] | 0.24 [0.12–0.54] | 0.25 [0.13–0.55] | 1.00 |
| enetw | prox_incopy | short | 14 | 0.96 [0.89–1.04] | 0.93 [0.88–0.96] | 0.98 [0.89–1.05] | 0.68 [0.29–1.01] | 0.72 [0.32–1.08] | 1.00 |
| enetw | prox_incopy | medium | 14 | 0.85 [0.84–0.88] | 0.87 [0.84–1.00] | 0.85 [0.84–0.87] | 0.22 [0.14–0.33] | 0.24 [0.15–0.35] | 1.00 |
| enetw | prox_incopy | large | 12 | 0.87 [0.85–0.94] | 1.00 [1.00–1.00] | 0.87 [0.85–0.95] | 0.35 [0.17–0.82] | 0.37 [0.18–0.83] | 1.00 |
| enetw | prox_out | short | 14 | 0.97 [0.93–1.01] | 0.93 [0.86–0.97] | 0.99 [0.93–1.05] | 1.00 [0.97–1.06] | 1.03 [1.01–1.06] | 0.99 |
| enetw | prox_out | medium | 14 | 0.92 [0.89–0.95] | 0.93 [0.89–1.00] | 0.92 [0.90–0.93] | 0.92 [0.90–0.95] | 1.00 [1.00–1.01] | 1.00 |
| enetw | prox_out | large | 12 | 0.96 [0.92–1.00] | 1.00 [1.00–1.00] | 0.96 [0.92–1.00] | 0.96 [0.92–1.00] | 1.00 [1.00–1.00] | 1.00 |
| enetw | sparse_prox | large | 2 | 0.99 [0.98–0.99] | 0.99 [0.99–0.99] | 0.99 [0.99–0.99] | 0.99 [0.99–0.99] | 1.00 [1.00–1.00] | 1.00 |
| enetw | valprox_out | short | 14 | 0.99 [0.97–1.00] | 0.97 [0.92–1.01] | 1.01 [0.97–1.06] | 1.01 [0.98–1.03] | 1.03 [1.00–1.07] | 1.00 |
| enetw | valprox_out | medium | 14 | 0.95 [0.93–0.97] | 0.95 [0.92–1.00] | 0.95 [0.93–0.97] | 0.95 [0.93–0.98] | 1.00 [1.00–1.01] | 1.00 |
| enetw | valprox_out | large | 12 | 0.97 [0.95–1.00] | 1.00 [1.00–1.00] | 0.97 [0.95–1.00] | 0.97 [0.95–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1 | block_loop | short | 4 | 0.96 [0.90–1.01] | 0.96 [0.92–1.02] | 0.98 [0.94–1.02] | 0.96 [0.90–1.00] | 1.01 [1.00–1.04] | 1.06 |
| l1 | block_loop | medium | 4 | 0.79 [0.77–0.82] | 0.78 [0.77–0.81] | 0.78 [0.77–0.81] | 0.78 [0.77–0.79] | 1.00 [1.00–1.00] | 1.00 |
| l1 | prox_alloc | short | 14 | 0.99 [0.96–1.01] | 1.00 [0.98–1.04] | 1.00 [0.97–1.04] | 1.00 [0.97–1.03] | 1.01 [0.99–1.04] | 0.99 |
| l1 | prox_alloc | medium | 14 | 1.00 [0.96–1.02] | 1.00 [0.97–1.02] | 1.00 [0.96–1.03] | 1.01 [0.99–1.03] | 1.00 [0.99–1.01] | 1.00 |
| l1 | prox_alloc | large | 12 | 1.03 [1.01–1.05] | 1.00 [0.98–1.01] | 1.03 [1.01–1.05] | 1.02 [1.00–1.04] | 1.00 [0.99–1.01] | 1.01 |
| l1 | prox_g_out | short | 14 | 1.01 [0.97–1.08] | 0.98 [0.95–1.04] | 1.01 [0.99–1.06] | 1.04 [1.01–1.14] | 1.05 [0.99–1.13] | 0.93 |
| l1 | prox_g_out | medium | 14 | 1.00 [0.99–1.00] | 0.97 [0.93–1.00] | 1.00 [0.99–1.00] | 1.01 [1.00–1.02] | 1.08 [1.02–1.14] | 0.89 |
| l1 | prox_g_out | large | 12 | 1.00 [1.00–1.00] | 0.97 [0.92–1.00] | 1.00 [1.00–1.00] | 1.01 [0.99–1.02] | 1.08 [1.02–1.14] | 0.89 |
| l1 | prox_in | short | 14 | 0.89 [0.74–1.05] | 0.91 [0.75–1.10] | 0.93 [0.75–1.18] | 0.58 [0.18–1.05] | 0.62 [0.20–1.26] | 1.06 |
| l1 | prox_in | medium | 14 | 0.81 [0.80–0.83] | 0.84 [0.78–1.06] | 0.81 [0.78–0.83] | 0.15 [0.10–0.25] | 0.20 [0.13–0.33] | 1.09 |
| l1 | prox_in | large | 12 | 0.80 [0.78–0.81] | 1.03 [0.99–1.08] | 0.80 [0.78–0.82] | 0.19 [0.12–0.40] | 0.24 [0.16–0.43] | 1.08 |
| l1 | prox_incopy | short | 14 | 0.89 [0.74–1.03] | 0.94 [0.76–1.14] | 0.96 [0.76–1.15] | 0.66 [0.25–1.03] | 0.67 [0.26–1.14] | 1.04 |
| l1 | prox_incopy | medium | 14 | 0.84 [0.81–0.86] | 0.86 [0.80–1.05] | 0.84 [0.81–0.86] | 0.23 [0.13–0.39] | 0.26 [0.16–0.45] | 1.08 |
| l1 | prox_incopy | large | 12 | 0.86 [0.82–0.94] | 1.02 [1.00–1.06] | 0.86 [0.83–0.94] | 0.36 [0.20–0.83] | 0.40 [0.24–0.84] | 1.07 |
| l1 | prox_out | short | 14 | 0.95 [0.83–1.05] | 0.95 [0.81–1.10] | 0.96 [0.83–1.10] | 0.94 [0.83–1.12] | 1.05 [0.99–1.14] | 0.97 |
| l1 | prox_out | medium | 14 | 0.77 [0.76–0.80] | 0.80 [0.76–1.00] | 0.78 [0.76–0.82] | 0.77 [0.75–0.82] | 1.00 [1.00–1.03] | 1.00 |
| l1 | prox_out | large | 12 | 0.86 [0.76–1.01] | 1.00 [1.00–1.00] | 0.86 [0.76–1.01] | 0.86 [0.77–1.01] | 1.00 [1.00–1.00] | 1.00 |
| l1 | sparse_prox | large | 2 | 0.98 [0.96–1.00] | 0.98 [0.96–1.00] | 0.98 [0.96–1.00] | 0.98 [0.96–1.00] | 1.00 [1.00–1.00] | 1.01 |
| l1 | valprox_out | short | 14 | 0.97 [0.90–1.04] | 0.98 [0.90–1.03] | 0.99 [0.91–1.06] | 0.98 [0.90–1.05] | 1.02 [0.98–1.07] | 0.99 |
| l1 | valprox_out | medium | 14 | 0.85 [0.83–0.90] | 0.87 [0.81–1.00] | 0.85 [0.83–0.90] | 0.85 [0.82–0.91] | 1.00 [1.00–1.01] | 1.00 |
| l1 | valprox_out | large | 12 | 0.90 [0.83–1.00] | 1.00 [1.00–1.00] | 0.90 [0.83–1.00] | 0.90 [0.83–1.00] | 1.00 [1.00–1.00] | 1.00 |
| l1w | block_loop | short | 4 | 0.97 [0.96–0.98] | 0.95 [0.92–0.98] | 0.95 [0.91–0.97] | 0.97 [0.94–1.02] | 0.98 [0.93–1.00] | 0.99 |
| l1w | block_loop | medium | 4 | 0.86 [0.84–0.88] | 0.86 [0.84–0.88] | 0.86 [0.84–0.88] | 0.87 [0.84–0.90] | 1.00 [1.00–1.00] | 1.00 |
| l1w | prox_alloc | short | 14 | 1.00 [0.99–1.01] | 1.00 [0.99–1.01] | 0.99 [0.98–1.01] | 1.00 [0.97–1.02] | 1.01 [1.00–1.02] | 1.00 |
| l1w | prox_alloc | medium | 14 | 1.00 [0.98–1.02] | 1.00 [0.98–1.01] | 1.00 [0.99–1.01] | 1.00 [0.98–1.01] | 1.00 [0.99–1.02] | 1.00 |
| l1w | prox_alloc | large | 12 | 1.01 [1.00–1.01] | 1.00 [0.99–1.01] | 1.01 [1.00–1.02] | 1.00 [0.99–1.02] | 1.00 [0.99–1.00] | 1.00 |
| l1w | prox_g_out | short | 14 | 0.99 [0.91–1.15] | 0.98 [0.91–1.13] | 0.99 [0.91–1.13] | 1.00 [0.91–1.17] | 1.06 [1.00–1.12] | 1.11 |
| l1w | prox_g_out | medium | 14 | 0.91 [0.90–0.92] | 0.91 [0.90–0.92] | 0.91 [0.90–0.92] | 0.91 [0.90–0.92] | 1.05 [1.00–1.11] | 1.13 |
| l1w | prox_g_out | large | 12 | 0.92 [0.91–0.98] | 0.92 [0.90–0.98] | 0.92 [0.91–0.98] | 0.92 [0.90–0.98] | 1.04 [1.00–1.11] | 1.11 |
| l1w | prox_in | short | 14 | 0.94 [0.87–1.04] | 0.94 [0.79–1.03] | 0.94 [0.87–1.09] | 0.55 [0.16–1.07] | 0.58 [0.19–1.05] | 1.00 |
| l1w | prox_in | medium | 14 | 0.84 [0.83–0.85] | 0.87 [0.84–1.00] | 0.83 [0.82–0.84] | 0.17 [0.11–0.27] | 0.19 [0.13–0.30] | 1.00 |
| l1w | prox_in | large | 12 | 0.85 [0.83–0.89] | 1.00 [1.00–1.00] | 0.85 [0.82–0.89] | 0.28 [0.14–0.64] | 0.29 [0.15–0.65] | 1.00 |
| l1w | prox_incopy | short | 14 | 0.95 [0.83–1.04] | 0.98 [0.85–1.09] | 0.98 [0.87–1.17] | 0.62 [0.25–1.03] | 0.64 [0.27–1.05] | 0.99 |
| l1w | prox_incopy | medium | 14 | 0.85 [0.84–0.87] | 0.88 [0.85–1.00] | 0.85 [0.84–0.86] | 0.23 [0.14–0.37] | 0.25 [0.16–0.39] | 1.00 |
| l1w | prox_incopy | large | 12 | 0.88 [0.85–0.95] | 1.00 [1.00–1.00] | 0.87 [0.84–0.95] | 0.40 [0.20–0.86] | 0.41 [0.21–0.87] | 1.00 |
| l1w | prox_out | short | 14 | 0.97 [0.89–1.04] | 0.99 [0.89–1.05] | 0.99 [0.90–1.07] | 0.98 [0.91–1.07] | 1.03 [0.99–1.09] | 1.00 |
| l1w | prox_out | medium | 14 | 0.88 [0.84–0.92] | 0.90 [0.83–1.00] | 0.88 [0.83–0.92] | 0.88 [0.82–0.92] | 1.00 [1.00–1.01] | 1.00 |
| l1w | prox_out | large | 12 | 0.96 [0.90–1.01] | 1.00 [1.00–1.00] | 0.96 [0.90–1.01] | 0.96 [0.90–1.01] | 1.00 [1.00–1.00] | 1.00 |
| l1w | sparse_prox | large | 2 | 1.00 [0.99–1.00] | 0.99 [0.99–1.00] | 1.00 [0.99–1.00] | 1.00 [0.99–1.00] | 1.00 [0.99–1.01] | 1.00 |
| l1w | valprox_out | short | 14 | 0.98 [0.93–1.00] | 0.99 [0.94–1.02] | 1.01 [0.94–1.05] | 1.00 [0.94–1.02] | 1.02 [0.99–1.04] | 1.00 |
| l1w | valprox_out | medium | 14 | 0.93 [0.90–0.95] | 0.93 [0.90–1.00] | 0.93 [0.90–0.95] | 0.93 [0.90–0.95] | 1.00 [1.00–1.01] | 1.00 |
| l1w | valprox_out | large | 12 | 0.97 [0.94–1.00] | 1.00 [1.00–1.00] | 0.97 [0.94–1.00] | 0.97 [0.94–1.00] | 1.00 [1.00–1.00] | 1.00 |

A3M4: 0 of 1000 cells slower in every process, 227 faster in every process, 12 above 1.05 in some processes only, 212 undecided (control unclean)
  some    enetw/prox_alloc/Float64/n8/normal           0.970–1.102   44.7 -> 44.3 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    l1/prox_out/Float32/n65536/normal            0.757–1.072   9762.0 -> 7488.7 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]
  some    l1w/prox_g_out/Float32/n1/normal             1.000–1.065   8.4 -> 8.4 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1/valprox_out/Float32/n65536/normal         0.836–1.065   13794.4 -> 11551.2 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]
  some    l1/prox_alloc/Float32/n262144/normal         1.021–1.060   130056.8 -> 135139.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n2/normal            0.979–1.059   37.4 -> 36.5 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n32768/normal        1.034–1.056   11981.7 -> 12557.8 ns   output offsets [1472, 896, 128, 3712, 3648, 896]
  some    l1/prox_alloc/Float64/n2048/normal           1.013–1.054   1855.6 -> 1906.8 ns   output offsets [3840, 0, 768, 1152, 2048, 3840]
  some    enet/prox_alloc/Float64/n16384/normal        0.989–1.053   12140.8 -> 12728.2 ns   output offsets [1408, 896, 3968, 960, 1728, 64]
  some    l1/prox_alloc/Float64/n16384/normal          1.007–1.053   12140.1 -> 12652.7 ns   output offsets [1408, 896, 3968, 960, 1728, 64]
  some    enet/prox_out/Float32/n65536/normal          0.811–1.053   10850.4 -> 8862.5 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]
  some    enetw/prox_alloc/Float64/n2/normal           0.989–1.053   40.3 -> 40.8 ns   output offsets [48, 48, 48, 48, 48, 48]

A3c: 1 of 1000 cells slower in every process, 123 faster in every process, 6 above 1.05 in some processes only, 156 undecided (control unclean)
  slower  l1/prox_in/Float64/n1/normal                 1.095–1.095   7.0 -> 7.7 ns
  some    l1w/prox_g_out/Float32/n1/normal             1.000–1.080   8.4 -> 8.4 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1/prox_incopy/Float64/n1/normal             1.016–1.074   11.0 -> 11.6 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_alloc/Float32/n1/normal              1.017–1.069   27.0 -> 28.0 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1/prox_alloc/Float32/n2/normal              0.995–1.064   28.2 -> 28.4 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1w/prox_alloc/Float64/n8/normal             0.992–1.063   34.0 -> 34.4 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    enet/prox_alloc/Float32/n2/normal            0.981–1.053   37.4 -> 36.6 ns   output offsets [64, 96, 64, 64, 64, 64]

A3cr: 2 of 1000 cells slower in every process, 222 faster in every process, 17 above 1.05 in some processes only, 211 undecided (control unclean)
  slower  l1/prox_in/Float64/n1/normal                 1.143–1.191   7.0 -> 8.3 ns
  slower  l1/prox_incopy/Float64/n1/normal             1.091–1.181   11.0 -> 12.4 ns
  some    enetw/prox_alloc/Float64/n1/normal           1.001–1.119   38.9 -> 39.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_alloc/Float64/n4/normal           0.991–1.095   41.5 -> 42.0 ns   output offsets [64, 64, 64, 64, 64, 128]
  some    enetw/prox_alloc/Float64/n2/normal           0.979–1.092   40.3 -> 40.8 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    l1/prox_alloc/Float32/n1/normal              1.024–1.086   27.0 -> 27.9 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enetw/prox_alloc/Float64/n8/normal           0.985–1.078   44.7 -> 44.9 ns   output offsets [96, 96, 96, 96, 96, 96]
  some    l1/prox_alloc/Float32/n2/normal              0.998–1.072   28.2 -> 28.6 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1w/valprox_out/Float64/n1/normal            1.033–1.071   14.9 -> 15.5 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_g_out/Float32/n1/normal             1.000–1.066   8.4 -> 8.4 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1w/valprox_out/Float32/n1/normal            1.031–1.065   15.0 -> 15.7 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enet/prox_alloc/Float32/n32768/normal        1.035–1.064   11981.7 -> 12596.9 ns   output offsets [1472, 896, 128, 3712, 3648, 896]
  some    l1/valprox_out/Float32/n65536/normal         0.835–1.064   13794.4 -> 11541.8 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]
  some    l1/prox_out/Float32/n65536/normal            0.766–1.063   9762.0 -> 7510.0 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]
  some    enet/prox_alloc/Float32/n2/normal            0.994–1.063   37.4 -> 37.5 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    enet/prox_alloc/Float64/n16384/normal        1.007–1.063   12140.8 -> 12580.5 ns   output offsets [1408, 896, 3968, 960, 1728, 64]
  some    enet/prox_out/Float32/n65536/normal          0.812–1.054   10850.4 -> 8872.6 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]
  some    enet/prox_alloc/Float64/n2048/normal         1.008–1.051   1871.6 -> 1929.8 ns   output offsets [3840, 0, 768, 1152, 2048, 3840]
  some    l1w/prox_out/Float32/n65536/normal           0.910–1.050   11956.4 -> 10973.6 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]

A3i: 0 of 1000 cells slower in every process, 230 faster in every process, 20 above 1.05 in some processes only, 238 undecided (control unclean)
  some    enetw/prox_alloc/Float64/n1/normal           1.008–1.126   38.9 -> 39.5 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1w/prox_g_out/Float32/n1/normal             1.040–1.120   8.4 -> 9.0 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1/prox_in/Float64/n1/normal                 1.048–1.095   7.0 -> 7.4 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_out/Float32/n65536/normal            0.764–1.090   9762.0 -> 7491.3 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]
  some    enet/prox_alloc/Float64/n16384/normal        1.016–1.069   12140.8 -> 12899.3 ns   output offsets [1408, 896, 3968, 960, 1728, 64]
  some    enet/prox_g_out/Float32/n1048576/normal      1.047–1.064   1169223.5 -> 1244196.5 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/valprox_out/Float32/n65536/normal         0.831–1.064   13794.4 -> 11494.0 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]
  some    enet/prox_g_out/Float64/n16/normal           1.021–1.063   49.6 -> 52.5 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    enetw/prox_g_out/Float32/n128/normal         1.047–1.063   116.6 -> 123.0 ns   output offsets [544, 544, 544, 544, 544, 544]
  some    enet/prox_g_out/Float32/n1024/normal         1.022–1.058   800.1 -> 819.0 ns   output offsets [128, 64, 3392, 64, 832, 2880]
  some    enet/prox_g_out/Float32/n262144/normal       1.044–1.057   292851.1 -> 306500.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_g_out/Float32/n32768/normal       1.043–1.056   25180.3 -> 26264.2 ns   output offsets [1472, 896, 128, 3712, 3648, 896]
  some    enetw/prox_g_out/Float32/n256/normal         1.046–1.054   212.8 -> 223.5 ns   output offsets [1088, 1088, 1088, 1088, 1088, 1088]
  some    l1w/prox_out/Float32/n65536/normal           0.913–1.054   11956.4 -> 11000.8 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]
  some    enet/prox_alloc/Float32/n32768/normal        1.036–1.054   11981.7 -> 12525.1 ns   output offsets [1472, 896, 128, 3712, 3648, 896]
  some    enet/prox_alloc/Float64/n4096/normal         1.005–1.053   3363.9 -> 3475.0 ns   output offsets [1408, 128, 2560, 3648, 1088, 2432]
  some    enetw/prox_g_out/Float32/n262144/normal      1.047–1.053   310984.9 -> 326274.0 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_out/Float32/n65536/normal          0.804–1.053   10850.4 -> 8834.6 ns   output offsets [3136, 128, 3264, 4032, 2304, 3968]
  some    enet/prox_g_out/Float32/n64/normal           1.025–1.051   65.0 -> 67.7 ns   output offsets [288, 288, 288, 288, 288, 288]
  some    enetw/prox_g_out/Float32/n1024/normal        1.046–1.051   808.0 -> 846.1 ns   output offsets [128, 64, 3392, 64, 832, 2880]

Mi: 5 of 1000 cells slower in every process, 121 faster in every process, 16 above 1.05 in some processes only, 163 undecided (control unclean)
  slower  l1/prox_in/Float64/n1/normal                 1.095–1.381   7.0 -> 8.9 ns
  slower  l1/prox_incopy/Float64/n1/normal             1.066–1.212   11.0 -> 12.7 ns
  slower  enet/prox_g_out/Float64/n16/normal           1.055–1.063   49.6 -> 52.5 ns
  slower  enetw/prox_g_out/Float32/n128/normal         1.053–1.067   116.6 -> 123.4 ns
  slower  enetw/prox_g_out/Float32/n256/normal         1.051–1.055   212.8 -> 223.9 ns
  some    enetw/valprox_out/Float64/n64/normal         0.997–1.142   62.6 -> 64.1 ns   output offsets [1088, 1088, 544, 1088, 1664, 1088]
  some    enet/prox_alloc/Float32/n8/normal            1.005–1.099   39.4 -> 40.2 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    l1/prox_out/Float32/n1024/normal             0.993–1.093   132.4 -> 138.0 ns   output offsets [128, 64, 3392, 64, 832, 2880]
  some    enetw/block_loop/Float32/n128/normal         0.985–1.085   34.0 -> 33.9 ns   output offsets [544, 544, 544, 544, 544, 544]
  some    l1w/valprox_out/Float32/n1/normal            1.000–1.082   15.0 -> 15.3 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1w/prox_g_out/Float32/n1/normal             1.000–1.080   8.4 -> 8.7 ns   output offsets [64, 96, 64, 64, 64, 64]
  some    l1w/prox_out/Float64/n16/normal              0.995–1.077   12.9 -> 13.2 ns   output offsets [160, 160, 160, 160, 160, 160]
  some    enetw/prox_alloc/Float64/n4/normal           1.004–1.065   41.5 -> 42.4 ns   output offsets [64, 64, 64, 64, 64, 128]
  some    enet/prox_g_out/Float32/n262144/normal       1.042–1.057   292851.1 -> 306279.7 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enet/prox_g_out/Float32/n1048576/normal      1.048–1.057   1169223.5 -> 1228700.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_g_out/Float32/n32768/normal       1.041–1.056   25180.3 -> 26261.9 ns   output offsets [1472, 896, 128, 3712, 3648, 896]
  some    enetw/prox_g_out/Float32/n1024/normal        1.045–1.054   808.0 -> 847.9 ns   output offsets [128, 64, 3392, 64, 832, 2880]
  some    l1/prox_alloc/Float64/n2/normal              0.992–1.052   27.7 -> 28.4 ns   output offsets [48, 48, 48, 48, 48, 48]
  some    enet/prox_g_out/Float32/n1024/normal         1.020–1.052   800.1 -> 816.8 ns   output offsets [128, 64, 3392, 64, 832, 2880]
  some    enetw/prox_g_out/Float32/n262144/normal      1.046–1.051   310984.9 -> 326284.8 ns   output offsets [64, 64, 64, 64, 64, 64]
  some    enetw/prox_g_out/Float32/n1000/normal        1.045–1.050   790.6 -> 829.7 ns   output offsets [0, 1792, 3136, 3392, 3520, 2304]
