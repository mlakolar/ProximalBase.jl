# Measured primary comparisons

R = post850/original; intervals resample paired process rounds. Values are ns per complete case.

| Case | Original ns | Post850 ns | R [95% CI] | Bytes old→new | Allocs old→new | A/A |
|---|---:|---:|---|---|---|---:|
| sparse/Float32/n10000/s100/b1/scalar/active_sweep | 63.0 | 70.0 | 1.121 [1.094, 1.135] | 0→0 | 0→0 | 1.000 |
| sparse/Float32/n10000/s100/b1/scalar/dot | 37.0 | 40.0 | 1.081 [1.054, 1.122] | 0→0 | 0→0 | 1.000 |
| sparse/Float32/n10000/s100/b1/scalar/matvec | 217.0 | 219.0 | 1.014 [1.005, 1.051] | 0→0 | 0→0 | 0.998 |
| sparse/Float32/n10000/s1000/b1/scalar/active_sweep | 553.0 | 578.0 | 1.047 [1.009, 1.144] | 0→0 | 0→0 | 0.999 |
| sparse/Float32/n10000/s1000/b1/scalar/cd_fixed | 2907.0 | 2982.5 | 1.022 [1.017, 1.032] | 0→0 | 0→0 | 0.997 |
| sparse/Float32/n1000000/s100/b1/scalar/dot | 51.0 | 56.0 | 1.100 [1.080, 1.125] | 0→0 | 0→0 | 1.000 |
| sparse/Float32/n1000000/s100/b1/scalar/read_scalar | 18.0 | 18.0 | 1.000 [1.000, 1.000] | 0→0 | 0→0 | 1.000 |
| sparse/Float32/n1000000/s100/b1/scalar/write_scalar | 18.0 | 18.0 | 1.000 [0.944, 1.056] | 0→0 | 0→0 | 1.000 |
| sparse/Float32/n1000000/s10000/b1/scalar/active_sweep | 6831.0 | 7003.0 | 1.021 [1.011, 1.046] | 0→0 | 0→0 | 1.002 |
| sparse/Float64/n10000/s100/b1/scalar/active_sweep | 63.0 | 69.5 | 1.113 [1.097, 1.121] | 0→0 | 0→0 | 0.992 |
| sparse/Float64/n10000/s100/b1/scalar/clear | 2613.0 | 15.0 | 0.006 [0.005, 0.006] | 0→0 | 0→0 | 0.990 |
| sparse/Float64/n10000/s100/b1/scalar/dot | 48.0 | 52.0 | 1.062 [1.058, 1.085] | 0→0 | 0→0 | 1.000 |
| sparse/Float64/n10000/s100/b1/scalar/write_scalar | 14.0 | 13.0 | 0.929 [0.929, 1.000] | 0→0 | 0→0 | 1.000 |
| sparse/Float64/n10000/s1000/b1/scalar/active_sweep | 610.0 | 601.0 | 0.998 [0.981, 1.073] | 0→0 | 0→0 | 1.003 |
| sparse/Float64/n10000/s1000/b1/scalar/matvec | 1920.0 | 2081.0 | 1.081 [1.076, 1.086] | 0→0 | 0→0 | 1.000 |
| sparse/Float64/n10000/s1000/b1/scalar/repeat_update | 3685.5 | 3837.5 | 1.042 [0.960, 1.073] | 0→0 | 0→0 | 1.047 |
| sparse/Float64/n10000/s10000/b1/scalar/dot | 2681.5 | 2829.5 | 1.054 [1.052, 1.057] | 0→0 | 0→0 | 0.997 |
| sparse/Float64/n10000/s10000/b1/scalar/matvec | 21182.0 | 22617.5 | 1.075 [1.032, 1.126] | 0→0 | 0→0 | 0.990 |
| sparse/Float64/n1000000/s100/b1/scalar/dot | 52.5 | 57.0 | 1.077 [1.061, 1.118] | 0→0 | 0→0 | 1.000 |
| sparse/Float64/n1000000/s100/b1/scalar/matvec | 207.0 | 213.0 | 1.024 [1.019, 1.039] | 0→0 | 0→0 | 1.000 |
| sparse/Float64/n1000000/s1000/b1/scalar/matvec | 2332.0 | 2518.0 | 1.082 [1.076, 1.092] | 0→0 | 0→0 | 0.998 |
| sparse/Float64/n1000000/s1000/b1/scalar/repeat_update | 4791.0 | 4649.5 | 0.987 [0.947, 1.015] | 0→0 | 0→0 | 1.004 |
| sparse/Float64/n1000000/s1000/b1/scalar/write_active | 763.0 | 739.0 | 0.988 [0.948, 1.003] | 0→0 | 0→0 | 1.002 |
| symmetric/Float32/n1414/s100/b1/triangle/dropzeros | 85.5 | 86.0 | 1.012 [1.000, 1.087] | 0→0 | 0→0 | 1.045 |
| symmetric/Float32/n1414/s100/b1/triangle/matmat | 1352.0 | 1485.0 | 1.084 [1.072, 1.112] | 0→0 | 0→0 | 1.004 |
| symmetric/Float32/n1414/s1000/b1/triangle/matmat | 10670.0 | 12512.5 | 1.171 [1.129, 1.210] | 0→0 | 0→0 | 1.016 |
| symmetric/Float32/n1414/s10000/b1/triangle/matmat | 99865.0 | 120011.5 | 1.149 [1.116, 1.220] | 0→0 | 0→0 | 1.057 |
| symmetric/Float32/n144/s100/b1/triangle/matmat | 937.0 | 1086.5 | 1.153 [1.142, 1.178] | 0→0 | 0→0 | 0.994 |
| symmetric/Float32/n144/s1000/b1/triangle/active_sweep | 608.0 | 572.0 | 0.942 [0.915, 1.018] | 0→0 | 0→0 | 0.993 |
| symmetric/Float32/n144/s1000/b1/triangle/matmat | 9244.0 | 10849.5 | 1.168 [1.166, 1.187] | 0→0 | 0→0 | 1.007 |
| symmetric/Float32/n144/s1000/b1/triangle/repeat_update | 3767.0 | 3385.5 | 0.927 [0.890, 1.023] | 0→0 | 0→0 | 0.995 |
| symmetric/Float32/n144/s1000/b1/triangle/write_active | 532.0 | 533.5 | 1.001 [0.962, 1.103] | 0→0 | 0→0 | 0.997 |
| symmetric/Float32/n144/s1000/b1/triangle/zero_active | 530.0 | 528.0 | 1.006 [0.972, 1.083] | 0→0 | 0→0 | 1.004 |
| symmetric/Float32/n144/s10000/b1/triangle/bilinear | 16592.5 | 32134.5 | 1.940 [1.921, 1.947] | 0→0 | 0→0 | 0.990 |
| symmetric/Float32/n144/s10000/b1/triangle/matmat | 94390.5 | 109397.5 | 1.158 [1.155, 1.248] | 0→0 | 0→0 | 1.009 |
| symmetric/Float64/n1414/s1000/b1/triangle/matmat | 13844.5 | 15060.5 | 1.090 [1.065, 1.114] | 0→0 | 0→0 | 0.997 |
| symmetric/Float64/n1414/s10000/b1/triangle/matmat | 127379.5 | 141125.5 | 1.100 [1.092, 1.122] | 0→0 | 0→0 | 0.994 |
| symmetric/Float64/n144/s100/b1/triangle/matmat | 1054.5 | 1181.5 | 1.033 [0.984, 1.187] | 0→0 | 0→0 | 1.010 |
| symmetric/Float64/n144/s100/b1/triangle/zero_active | 53.0 | 53.0 | 1.000 [0.981, 1.029] | 0→0 | 0→0 | 1.010 |
| symmetric/Float64/n144/s1000/b1/triangle/clear | 4829.0 | 91.5 | 0.018 [0.017, 0.019] | 0→0 | 0→0 | 1.012 |
| symmetric/Float64/n144/s1000/b1/triangle/matmat | 9419.5 | 11198.0 | 1.205 [1.164, 1.342] | 0→0 | 0→0 | 1.000 |
| symmetric/Float64/n144/s10000/b1/triangle/bilinear | 16554.5 | 32134.0 | 1.940 [1.919, 1.965] | 0→0 | 0→0 | 0.992 |
| symmetric/Float64/n144/s10000/b1/triangle/matmat | 94676.0 | 110622.5 | 1.169 [1.160, 1.204] | 0→0 | 0→0 | 1.000 |
