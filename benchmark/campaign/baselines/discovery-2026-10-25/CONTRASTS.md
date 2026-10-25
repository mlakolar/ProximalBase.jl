# Baseline contrasts

Ten paired repetitions within each Julia version. Times are arithmetic means of run means. Ratios use geometric within-repetition ratios and a t interval on log ratios. Rows compare different APIs/cache phases of the same package, not a before/after implementation change.

| Julia | Phase contrast | Type | Shape | Operation | Before / after (µs) | Paired ratio [95% interval] |
|---|---|---|---|---|---:|---:|
| julia-1.10.12 | construction+first / warmed prox | Float32 | 1024 | least_squares/prox! | 8952.09 / 16.98 | 528.57 [497.20, 561.91] |
| julia-1.10.12 | construction+first / warmed prox | Float32 | 128 | least_squares/prox! | 272.04 / 1.09 | 249.83 [242.89, 256.96] |
| julia-1.10.12 | construction+first / warmed prox | Float64 | 1024 | least_squares/prox! | 14098.02 / 44.43 | 317.79 [301.13, 335.38] |
| julia-1.10.12 | construction+first / warmed prox | Float64 | 128 | least_squares/prox! | 284.95 / 1.58 | 180.94 [170.59, 191.93] |
| julia-1.10.12 | construction+first / warmed prox | Float32 | 256 | tall/prox! | 17532.29 / 5.41 | 3242.32 [3136.50, 3351.71] |
| julia-1.10.12 | construction+first / warmed prox | Float32 | 32 | tall/prox! | 180.41 / 0.20 | 910.17 [855.44, 968.41] |
| julia-1.10.12 | construction+first / warmed prox | Float64 | 256 | tall/prox! | 30473.89 / 14.51 | 2100.18 [2013.30, 2190.81] |
| julia-1.10.12 | construction+first / warmed prox | Float64 | 32 | tall/prox! | 227.38 / 0.26 | 857.22 [768.63, 956.01] |
| julia-1.10.12 | construction+first / warmed prox | Float32 | 1024 | wide/prox! | 8840.04 / 13.45 | 658.97 [627.37, 692.16] |
| julia-1.10.12 | construction+first / warmed prox | Float32 | 128 | wide/prox! | 259.62 / 0.36 | 718.21 [700.02, 736.88] |
| julia-1.10.12 | construction+first / warmed prox | Float64 | 1024 | wide/prox! | 13242.12 / 38.87 | 340.72 [319.97, 362.82] |
| julia-1.10.12 | construction+first / warmed prox | Float64 | 128 | wide/prox! | 269.09 / 0.80 | 339.87 [292.36, 395.12] |
| julia-1.10.12 | full / nearby-call partial | Float32 | 128x64 | nuclear/prox! | 509.57 / 512.31 | 0.99 [0.99, 1.00] |
| julia-1.10.12 | full / nearby-call partial | Float32 | 512x256 | nuclear/prox! | 11165.60 / 402.74 | 27.72 [27.19, 28.27] |
| julia-1.10.12 | full / nearby-call partial | Float64 | 128x64 | nuclear/prox! | 669.41 / 667.63 | 1.00 [0.99, 1.02] |
| julia-1.10.12 | full / nearby-call partial | Float64 | 512x256 | nuclear/prox! | 19460.91 / 1428.35 | 13.62 [13.51, 13.74] |
| julia-1.10.12 | full / nearby-call partial | Float32 | 128x128 | psd_rank4/prox! | 315.49 / 49.49 | 6.38 [6.22, 6.53] |
| julia-1.10.12 | full / nearby-call partial | Float32 | 512x512 | psd_rank4/prox! | 6187.32 / 561.11 | 11.03 [10.76, 11.30] |
| julia-1.10.12 | full / nearby-call partial | Float64 | 128x128 | psd_rank4/prox! | 472.55 / 227.14 | 2.08 [2.05, 2.11] |
| julia-1.10.12 | full / nearby-call partial | Float64 | 512x512 | psd_rank4/prox! | 10922.68 / 2232.58 | 4.89 [4.69, 5.09] |
| julia-1.10.12 | full / nearby-call partial | Float32 | 128x64 | rank4/prox! | 505.88 / 59.76 | 8.47 [8.30, 8.63] |
| julia-1.10.12 | full / nearby-call partial | Float32 | 512x256 | rank4/prox! | 11160.42 / 356.29 | 31.33 [30.97, 31.69] |
| julia-1.10.12 | full / nearby-call partial | Float64 | 128x64 | rank4/prox! | 664.80 / 204.62 | 3.25 [3.18, 3.32] |
| julia-1.10.12 | full / nearby-call partial | Float64 | 512x256 | rank4/prox! | 19557.90 / 1440.99 | 13.57 [13.47, 13.68] |
| julia-1.10.12 | full / nearby-call partial | Float32 | 128x128 | trace/prox! | 811.91 / 60.44 | 13.43 [13.20, 13.66] |
| julia-1.10.12 | full / nearby-call partial | Float32 | 512x512 | trace/prox! | 17764.54 / 602.80 | 29.48 [28.65, 30.33] |
| julia-1.10.12 | full / nearby-call partial | Float64 | 128x128 | trace/prox! | 1235.23 / 264.71 | 4.67 [4.62, 4.71] |
| julia-1.10.12 | full / nearby-call partial | Float64 | 512x512 | trace/prox! | 34571.53 / 2187.54 | 15.80 [15.50, 16.10] |
| julia-1.13.2 | construction+first / warmed prox | Float32 | 1024 | least_squares/prox! | 8842.46 / 20.23 | 442.92 [403.59, 486.08] |
| julia-1.13.2 | construction+first / warmed prox | Float32 | 128 | least_squares/prox! | 251.82 / 1.16 | 216.56 [206.67, 226.93] |
| julia-1.13.2 | construction+first / warmed prox | Float64 | 1024 | least_squares/prox! | 14074.94 / 44.67 | 317.09 [299.12, 336.15] |
| julia-1.13.2 | construction+first / warmed prox | Float64 | 128 | least_squares/prox! | 295.89 / 1.64 | 180.00 [167.70, 193.21] |
| julia-1.13.2 | construction+first / warmed prox | Float32 | 256 | tall/prox! | 18694.66 / 6.42 | 2957.07 [2611.92, 3347.84] |
| julia-1.13.2 | construction+first / warmed prox | Float32 | 32 | tall/prox! | 190.30 / 0.22 | 876.02 [788.51, 973.23] |
| julia-1.13.2 | construction+first / warmed prox | Float64 | 256 | tall/prox! | 32997.71 / 14.91 | 2209.92 [2112.38, 2311.98] |
| julia-1.13.2 | construction+first / warmed prox | Float64 | 32 | tall/prox! | 230.61 / 0.27 | 853.72 [794.96, 916.82] |
| julia-1.13.2 | construction+first / warmed prox | Float32 | 1024 | wide/prox! | 8809.79 / 15.25 | 582.23 [528.43, 641.51] |
| julia-1.13.2 | construction+first / warmed prox | Float32 | 128 | wide/prox! | 277.32 / 0.40 | 687.34 [584.44, 808.34] |
| julia-1.13.2 | construction+first / warmed prox | Float64 | 1024 | wide/prox! | 14178.09 / 40.02 | 357.63 [314.36, 406.85] |
| julia-1.13.2 | construction+first / warmed prox | Float64 | 128 | wide/prox! | 276.67 / 0.89 | 311.48 [291.92, 332.35] |
| julia-1.13.2 | full / nearby-call partial | Float32 | 128x64 | nuclear/prox! | 510.43 / 506.57 | 1.01 [0.99, 1.02] |
| julia-1.13.2 | full / nearby-call partial | Float32 | 512x256 | nuclear/prox! | 11947.03 / 417.31 | 28.67 [26.97, 30.47] |
| julia-1.13.2 | full / nearby-call partial | Float64 | 128x64 | nuclear/prox! | 688.69 / 705.72 | 0.98 [0.95, 1.00] |
| julia-1.13.2 | full / nearby-call partial | Float64 | 512x256 | nuclear/prox! | 20452.54 / 1520.16 | 13.46 [13.11, 13.82] |
| julia-1.13.2 | full / nearby-call partial | Float32 | 128x128 | psd_rank4/prox! | 323.65 / 50.68 | 6.40 [6.14, 6.67] |
| julia-1.13.2 | full / nearby-call partial | Float32 | 512x512 | psd_rank4/prox! | 6778.42 / 582.74 | 11.67 [10.90, 12.48] |
| julia-1.13.2 | full / nearby-call partial | Float64 | 128x128 | psd_rank4/prox! | 493.25 / 238.69 | 2.08 [1.97, 2.19] |
| julia-1.13.2 | full / nearby-call partial | Float64 | 512x512 | psd_rank4/prox! | 11560.04 / 2511.12 | 4.67 [4.25, 5.13] |
| julia-1.13.2 | full / nearby-call partial | Float32 | 128x64 | rank4/prox! | 498.06 / 58.79 | 8.47 [8.29, 8.66] |
| julia-1.13.2 | full / nearby-call partial | Float32 | 512x256 | rank4/prox! | 11672.99 / 371.32 | 31.50 [30.61, 32.42] |
| julia-1.13.2 | full / nearby-call partial | Float64 | 128x64 | rank4/prox! | 698.01 / 210.74 | 3.31 [3.22, 3.40] |
| julia-1.13.2 | full / nearby-call partial | Float64 | 512x256 | rank4/prox! | 20756.77 / 1526.53 | 13.62 [12.95, 14.33] |
| julia-1.13.2 | full / nearby-call partial | Float32 | 128x128 | trace/prox! | 857.24 / 64.77 | 13.40 [12.12, 14.81] |
| julia-1.13.2 | full / nearby-call partial | Float32 | 512x512 | trace/prox! | 19436.39 / 624.32 | 31.10 [30.46, 31.75] |
| julia-1.13.2 | full / nearby-call partial | Float64 | 128x128 | trace/prox! | 1257.19 / 276.07 | 4.56 [4.43, 4.69] |
| julia-1.13.2 | full / nearby-call partial | Float64 | 512x512 | trace/prox! | 36119.85 / 2403.62 | 15.09 [14.40, 15.82] |

## Representative expensive warmed operators

| Julia | Case | Mean ± SD (µs) | 95% interval (µs) | Maximum bytes |
|---|---|---:|---:|---:|
| julia-1.13.2 | `separable/KLDivergence/value_and_prox!/Float64/100000/dense/warm` | 11401.52 ± 585.42 | [10982.76, 11820.27] | 0 |
| julia-1.10.12 | `separable/NegEntropy/value_and_prox!/Float64/100000/dense/warm` | 11025.46 ± 306.49 | [10806.23, 11244.70] | 0 |
| julia-1.10.12 | `separable/KLDivergence/value_and_prox!/Float64/100000/dense/warm` | 10685.75 ± 261.76 | [10498.52, 10872.99] | 0 |
| julia-1.13.2 | `separable/NegEntropy/value_and_prox!/Float64/100000/dense/warm` | 10685.11 ± 547.51 | [10293.47, 11076.74] | 0 |
| julia-1.10.12 | `separable/NegEntropy/prox!/Float64/100000/dense/warm` | 10271.91 ± 295.43 | [10060.58, 10483.24] | 0 |
| julia-1.13.2 | `separable/KLDivergence/prox!/Float64/100000/dense/warm` | 10188.06 ± 509.15 | [9823.87, 10552.26] | 0 |
| julia-1.13.2 | `separable/NegEntropy/prox!/Float64/100000/dense/warm` | 9796.87 ± 484.43 | [9450.35, 10143.38] | 0 |
| julia-1.13.2 | `separable/NormLq/value_and_prox!/Float64/100000/dense/warm` | 9586.17 ± 472.85 | [9247.94, 9924.41] | 0 |
| julia-1.10.12 | `separable/KLDivergence/prox!/Float64/100000/dense/warm` | 9475.00 ± 231.58 | [9309.35, 9640.65] | 0 |
| julia-1.10.12 | `separable/NormLq/value_and_prox!/Float64/100000/dense/warm` | 8141.94 ± 198.31 | [8000.08, 8283.79] | 0 |
| julia-1.13.2 | `separable/NormLq/prox!/Float64/100000/dense/warm` | 7664.71 ± 361.64 | [7406.03, 7923.39] | 0 |
| julia-1.13.2 | `ordered/SLOPE/prox!/Float64/100000/dense/warm` | 6759.08 ± 1078.99 | [5987.27, 7530.89] | 0 |
| julia-1.13.2 | `ordered/SLOPE/value_and_prox!/Float64/100000/dense/warm` | 6730.64 ± 1033.72 | [5991.21, 7470.07] | 0 |
| julia-1.10.12 | `ordered/SLOPE/prox!/Float64/100000/dense/warm` | 6622.72 ± 248.80 | [6444.75, 6800.69] | 0 |
| julia-1.10.12 | `ordered/SLOPE/value_and_prox!/Float64/100000/dense/warm` | 6593.52 ± 232.53 | [6427.19, 6759.85] | 0 |
| julia-1.10.12 | `separable/NormLq/prox!/Float64/100000/dense/warm` | 6576.49 ± 165.65 | [6458.00, 6694.99] | 0 |
| julia-1.13.2 | `logdet/positive_definite/prox!/Float64/256x256/dense/warm` | 6555.45 ± 554.71 | [6158.66, 6952.24] | 0 |
| julia-1.13.2 | `logdet/positive_definite/value_and_prox!/Float64/256x256/dense/warm` | 6541.57 ± 592.12 | [6118.03, 6965.12] | 0 |
| julia-1.10.12 | `logdet/positive_definite/value_and_prox!/Float64/256x256/dense/warm` | 6428.20 ± 144.41 | [6324.90, 6531.49] | 0 |
| julia-1.10.12 | `logdet/positive_definite/prox!/Float64/256x256/dense/warm` | 6426.79 ± 157.82 | [6313.91, 6539.68] | 0 |
| julia-1.10.12 | `separable/LogPenalty/value_and_prox!/Float64/100000/dense/warm` | 3148.17 ± 68.08 | [3099.48, 3196.87] | 0 |
| julia-1.13.2 | `separable/LogPenalty/value_and_prox!/Float64/100000/dense/warm` | 3139.46 ± 162.58 | [3023.16, 3255.76] | 0 |
| julia-1.13.2 | `logdet/positive_definite/hessian_vec!/Float64/256x256/dense/warm` | 2412.80 ± 205.78 | [2265.60, 2560.00] | 0 |
| julia-1.10.12 | `logdet/positive_definite/hessian_vec!/Float64/256x256/dense/warm` | 2343.03 ± 53.70 | [2304.62, 2381.44] | 0 |
| julia-1.10.12 | `ordered/FusedLasso/value_and_prox!/Float64/100000/dense/warm` | 1846.92 ± 43.64 | [1815.70, 1878.14] | 0 |
| julia-1.13.2 | `separable/NegLog/value_and_prox!/Float64/100000/dense/warm` | 1833.68 ± 79.72 | [1776.66, 1890.71] | 0 |
| julia-1.10.12 | `separable/NegLog/value_and_prox!/Float64/100000/dense/warm` | 1775.97 ± 46.56 | [1742.67, 1809.28] | 0 |
| julia-1.10.12 | `ordered/TotalVariation1D/value_and_prox!/Float64/100000/dense/warm` | 1742.82 ± 46.26 | [1709.73, 1775.91] | 0 |
| julia-1.10.12 | `ordered/FusedLasso/prox!/Float64/100000/dense/warm` | 1695.77 ± 57.76 | [1654.46, 1737.09] | 0 |
| julia-1.10.12 | `separable/LogPenalty/prox!/Float64/100000/dense/warm` | 1670.18 ± 43.59 | [1639.00, 1701.35] | 0 |
| julia-1.13.2 | `separable/LogPenalty/prox!/Float64/100000/dense/warm` | 1667.41 ± 78.94 | [1610.94, 1723.87] | 0 |
| julia-1.10.12 | `ordered/TotalVariation1D/prox!/Float64/100000/dense/warm` | 1656.09 ± 49.98 | [1620.33, 1691.84] | 0 |
| julia-1.10.12 | `ordered/IndIsotonic/prox!/Float64/100000/dense/warm` | 1619.49 ± 69.97 | [1569.43, 1669.54] | 0 |
| julia-1.10.12 | `ordered/IndIsotonic/value_and_prox!/Float64/100000/dense/warm` | 1596.01 ± 49.22 | [1560.80, 1631.22] | 0 |
| julia-1.13.2 | `ordered/FusedLasso/value_and_prox!/Float64/100000/dense/warm` | 1572.96 ± 80.53 | [1515.36, 1630.57] | 0 |
| julia-1.13.2 | `ordered/IndIsotonic/value_and_prox!/Float64/100000/dense/warm` | 1543.86 ± 96.89 | [1474.55, 1613.17] | 0 |
| julia-1.13.2 | `ordered/IndIsotonic/prox!/Float64/100000/dense/warm` | 1538.08 ± 76.14 | [1483.61, 1592.54] | 0 |
| julia-1.13.2 | `ordered/TotalVariation1D/value_and_prox!/Float64/100000/dense/warm` | 1455.01 ± 59.22 | [1412.65, 1497.37] | 0 |
| julia-1.13.2 | `ordered/FusedLasso/prox!/Float64/100000/dense/warm` | 1421.82 ± 103.67 | [1347.66, 1495.97] | 0 |
| julia-1.13.2 | `ordered/TotalVariation1D/prox!/Float64/100000/dense/warm` | 1382.61 ± 71.20 | [1331.67, 1433.54] | 0 |
| julia-1.10.12 | `constrained/scad_box/value_and_prox!/Float64/100000/dense/warm` | 1163.73 ± 30.43 | [1141.97, 1185.50] | 0 |
| julia-1.10.12 | `constrained/scad_box/prox!/Float64/100000/dense/warm` | 1113.50 ± 28.70 | [1092.97, 1134.03] | 0 |
| julia-1.13.2 | `logdet/positive_definite/quadratic_model!/Float64/256x256/dense/warm` | 1090.09 ± 83.48 | [1030.38, 1149.80] | 0 |
| julia-1.10.12 | `logdet/positive_definite/quadratic_model!/Float64/256x256/dense/warm` | 1055.15 ± 27.11 | [1035.76, 1074.55] | 0 |
| julia-1.13.2 | `logdet/positive_definite/value_and_gradient!/Float64/256x256/dense/warm` | 1040.87 ± 117.93 | [956.51, 1125.23] | 0 |
