Strata (processes): xeon2640v4-julia1.10.13 (3), xeon2640v4-julia1.13.4 (3), xeon4116-julia1.13.4 (3)

| tree | case group | xeon2640v4-julia1.10.13 | xeon2640v4-julia1.13.4 | xeon4116-julia1.13.4 |
|:--|:--|:--|:--|:--|
| A | l1/prox/Float64 (6) | 0.85 [0.67–0.99] | 0.87 [0.82–1.14] | 0.94 [0.83–1.02] |
| A | l1/prox_w/Float64 (6) | 0.84 [0.72–0.97] | 0.89 [0.81–1.00] | 0.93 [0.86–1.00] |
| A | l1/prox_g/Float64 (6) | 0.53 [0.26–0.79] | 0.87 [0.86–0.88] | 0.90 [0.70–0.94] |
| A | l1/prox_gw/Float64 (6) | 0.52 [0.26–0.77] | 0.87 [0.87–0.88] | 0.86 [0.84–0.95] |
| A | l1/valprox/Float64 (6) | 0.89 [0.77–0.97] | 0.91 [0.86–0.99] | 0.95 [0.85–1.00] |
| A | l1/valprox_w/Float64 (6) | 0.95 [0.89–0.99] | 0.94 [0.90–1.00] | 0.96 [0.90–1.00] |
| A | l1/prox/Float32 (6) | 0.96 [0.71–2.74] | 0.87 [0.79–1.00] | 0.89 [0.76–1.03] |
| A | l1/prox_w/Float32 (6) | 0.78 [0.68–0.99] | 0.87 [0.78–1.00] | 0.92 [0.87–1.00] |
| A | l1/prox_g/Float32 (6) | 0.52 [0.25–0.79] | 0.87 [0.86–0.88] | 0.89 [0.70–0.91] |
| A | l1/prox_gw/Float32 (6) | 0.51 [0.24–0.76] | 0.87 [0.87–0.87] | 0.90 [0.74–0.91] |
| A | l1/valprox/Float32 (6) | 0.90 [0.75–1.39] | 0.90 [0.85–1.00] | 0.93 [0.88–1.00] |
| A | l1/valprox_w/Float32 (6) | 0.96 [0.92–1.00] | 0.93 [0.88–1.00] | 0.95 [0.91–1.00] |
| B | l1/prox_g/Float64 (6) | 0.26 [0.13–0.60] | 0.39 [0.32–0.67] | 0.36 [0.20–0.96] |
| B | l1/prox_gw/Float64 (6) | 0.30 [0.15–0.64] | 0.46 [0.32–0.94] | 0.35 [0.23–1.00] |
| B | l1/prox_g/Float32 (6) | 0.11 [0.04–0.59] | 0.16 [0.11–0.40] | 0.19 [0.09–0.46] |
| B | l1/prox_gw/Float32 (6) | 0.12 [0.05–0.58] | 0.17 [0.11–0.41] | 0.19 [0.09–0.47] |
| C | l1/value/Float64 (6) | 0.63 [0.47–1.00] | 0.71 [0.61–1.00] | 0.80 [0.63–1.00] |
| C | l1/value/Float32 (6) | 0.55 [0.30–1.00] | 0.67 [0.51–1.00] | 0.72 [0.49–1.00] |
| C2 | l2/value/Float64 (14) | 0.72 [0.34–1.16] | 0.76 [0.48–0.98] | 0.78 [0.49–1.02] |
| C2 | l2/value/Float32 (14) | 0.54 [0.24–1.14] | 0.67 [0.26–0.96] | 0.72 [0.47–0.96] |
| D | l2/prox/Float64 (14) | 0.91 [0.71–1.10] | 0.92 [0.71–1.28] | 0.91 [0.71–1.11] |
| D | l2/valprox/Float64 (14) | 0.90 [0.71–1.14] | 0.92 [0.71–1.05] | 0.88 [0.71–1.06] |
| D | group/prox/Float64 (12) | 0.85 [0.75–0.93] | 0.89 [0.83–0.95] | 0.86 [0.74–0.95] |
| D | group/valprox/Float64 (12) | 0.87 [0.80–0.96] | 0.90 [0.80–0.97] | 0.85 [0.71–0.97] |
| D | group/block_sweep/Float64 (6) | 0.91 [0.87–0.98] | 0.91 [0.87–0.99] | 0.90 [0.87–1.05] |
| D | l2/prox/Float32 (14) | 0.91 [0.78–1.03] | 0.92 [0.78–1.01] | 0.86 [0.73–1.02] |
| D | l2/valprox/Float32 (14) | 0.91 [0.78–1.02] | 0.92 [0.78–1.02] | 0.90 [0.71–1.09] |
| D | group/prox/Float32 (12) | 0.84 [0.72–0.90] | 0.85 [0.75–0.94] | 0.83 [0.56–0.98] |
| D | group/valprox/Float32 (12) | 0.86 [0.78–0.92] | 0.84 [0.77–0.96] | 0.84 [0.76–1.05] |
| D | group/block_sweep/Float32 (6) | 0.90 [0.81–0.97] | 0.90 [0.83–0.97] | 0.88 [0.81–0.97] |
| F | group/block_sweep/Float64 (18) | 0.83 [0.71–1.03] | 0.78 [0.64–1.02] | 0.80 [0.67–1.02] |
| F | group/block_sweep/Float32 (18) | 0.79 [0.68–1.01] | 0.75 [0.62–1.03] | 0.76 [0.58–1.01] |
| P155 | l1/value_w/Float64 (6) | 0.23 [0.17–0.47] | 0.82 [0.70–1.00] | 1.16 [1.02–1.28] |
| P155 | l1/valprox_w/Float64 (6) | 0.45 [0.36–0.67] | 0.93 [0.87–1.04] | 1.06 [1.02–1.09] |
| P155 | l1/value_w/Float32 (6) | 0.11 [0.07–0.43] | 0.70 [0.56–0.97] | 0.97 [0.91–1.10] |
| P155 | l1/valprox_w/Float32 (6) | 0.27 [0.20–0.62] | 0.87 [0.81–0.99] | 0.98 [0.95–1.01] |

Cases slower than 1.05 in every process of the stratum with the control within [0.98, 1.02] in every process (and the cases faster than 0.95 likewise), over all 540 cases

| tree | xeon2640v4-julia1.10.13 | xeon2640v4-julia1.13.4 | xeon4116-julia1.13.4 |
|:--|:--|:--|:--|
| baseC | 0 slower, 0 faster (0 unresolved) | 0 slower, 0 faster (0 unresolved) | 0 slower, 0 faster (2 unresolved) |
| A | 1 slower, 41 faster (1 unresolved) | 0 slower, 54 faster (0 unresolved) | 0 slower, 29 faster (1 unresolved) |
| B | 0 slower, 16 faster (0 unresolved) | 0 slower, 26 faster (0 unresolved) | 0 slower, 6 faster (1 unresolved) |
| C | 0 slower, 8 faster (0 unresolved) | 0 slower, 8 faster (0 unresolved) | 0 slower, 8 faster (1 unresolved) |
| C2 | 2 slower, 17 faster (1 unresolved) | 0 slower, 17 faster (0 unresolved) | 0 slower, 16 faster (2 unresolved) |
| D | 0 slower, 65 faster (0 unresolved) | 0 slower, 48 faster (0 unresolved) | 0 slower, 37 faster (0 unresolved) |
| F | 0 slower, 20 faster (0 unresolved) | 0 slower, 16 faster (0 unresolved) | 0 slower, 5 faster (0 unresolved) |
| P155 | 0 slower, 23 faster (0 unresolved) | 0 slower, 16 faster (0 unresolved) | 8 slower, 1 faster (2 unresolved) |

Slower cases:
- `A` on xeon2640v4-julia1.10.13: l1/valprox/Float32/n16/normal 1.38–1.39
- `C2` on xeon2640v4-julia1.10.13: l2/value/Float32/n10/shrink 1.08–1.14; l2/value/Float32/n10/zero 1.08–1.14
- `P155` on xeon4116-julia1.13.4: l1/valprox_w/Float64/n128/normal 1.07–1.07; l1/value_w/Float64/n1024/normal 1.25–1.28; l1/valprox_w/Float64/n1024/normal 1.08–1.09; l1/value_w/Float64/n1024/zeroheavy 1.25–1.27; l1/valprox_w/Float64/n1024/zeroheavy 1.07–1.09; l1/value_w/Float64/n16384/normal 1.14–1.14; … (8 in all)
