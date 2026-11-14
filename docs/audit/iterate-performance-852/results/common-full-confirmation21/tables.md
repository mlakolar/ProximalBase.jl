# Measured primary comparisons

R = post850/original; intervals resample paired process rounds. Values are ns per complete case.

| Case | Original ns | Post850 ns | R [95% CI] | Bytes old→new | Allocs old→new | A/A |
|---|---:|---:|---|---|---|---:|
| sparse/Float32/n10000/s1000/b1/scalar/repeat_update | 3635.5 | 3758.5 | 0.986 [0.862, 1.059] | 0→0 | 0→0 | 0.951 |
| sparse/Float32/n10000/s10000/b1/scalar/clear | 4725.0 | 400.0 | 0.085 [0.081, 0.087] | 0→0 | 0→0 | 1.020 |
| sparse/Float32/n10000/s10000/b1/scalar/dot | 1917.0 | 2014.5 | 1.051 [1.049, 1.054] | 0→0 | 0→0 | 0.999 |
| sparse/Float32/n1000000/s100/b1/scalar/dot | 1702.5 | 1710.0 | 1.036 [0.941, 1.085] | 0→0 | 0→0 | 0.997 |
| sparse/Float32/n1000000/s100/b1/scalar/read_scalar | 302.0 | 317.5 | 1.066 [0.851, 1.287] | 0→0 | 0→0 | 1.035 |
| sparse/Float32/n1000000/s100/b1/scalar/write_active | 947.5 | 979.5 | 1.022 [0.976, 1.050] | 0→0 | 0→0 | 1.049 |
| sparse/Float32/n1000000/s100/b1/scalar/zero_active | 967.0 | 993.5 | 1.015 [0.928, 1.098] | 0→0 | 0→0 | 1.013 |
| sparse/Float64/n10000/s100/b1/scalar/matvec | 224.0 | 240.0 | 1.076 [1.071, 1.104] | 0→0 | 0→0 | 1.000 |
| sparse/Float64/n10000/s1000/b1/scalar/clear | 4705.5 | 90.0 | 0.019 [0.018, 0.020] | 0→0 | 0→0 | 1.005 |
| sparse/Float64/n10000/s1000/b1/scalar/matvec | 1926.0 | 2084.0 | 1.082 [1.080, 1.086] | 0→0 | 0→0 | 0.999 |
| sparse/Float64/n10000/s10000/b1/scalar/dot | 2563.5 | 2570.5 | 1.003 [1.000, 1.004] | 0→0 | 0→0 | 1.001 |
| sparse/Float64/n10000/s10000/b1/scalar/matvec | 21192.5 | 23584.0 | 1.108 [1.057, 1.133] | 0→0 | 0→0 | 1.007 |
| sparse/Float64/n1000000/s100/b1/scalar/read_scalar | 351.5 | 366.5 | 1.035 [0.875, 1.139] | 0→0 | 0→0 | 0.842 |
| sparse/Float64/n1000000/s100/b1/scalar/read_sequential | 1045.5 | 1108.0 | 1.060 [1.023, 1.083] | 0→0 | 0→0 | 1.020 |
| sparse/Float64/n1000000/s100/b1/scalar/repeat_update | 1401.5 | 1467.0 | 1.065 [0.971, 1.113] | 0→0 | 0→0 | 0.991 |
| sparse/Float64/n1000000/s100/b1/scalar/write_active | 1023.0 | 1058.0 | 1.024 [0.985, 1.109] | 0→0 | 0→0 | 1.025 |
| sparse/Float64/n1000000/s100/b1/scalar/write_scalar | 433.0 | 448.0 | 1.023 [0.982, 1.152] | 0→0 | 0→0 | 1.019 |
| sparse/Float64/n1000000/s1000/b1/scalar/active_sweep | 3197.0 | 3370.0 | 1.054 [1.030, 1.079] | 0→0 | 0→0 | 0.992 |
| sparse/Float64/n1000000/s1000/b1/scalar/read_scalar | 326.5 | 310.5 | 0.940 [0.888, 0.990] | 0→0 | 0→0 | 0.943 |
| sparse/Float64/n1000000/s10000/b1/scalar/active_sweep | 23264.5 | 24839.0 | 1.069 [1.026, 1.085] | 0→0 | 0→0 | 0.996 |
| sparse/Float64/n1000000/s10000/b1/scalar/read_scalar | 332.5 | 303.5 | 0.976 [0.865, 0.991] | 0→0 | 0→0 | 0.981 |
| symmetric/Float32/n1414/s100/b1/triangle/clear | 253531.0 | 210.5 | 0.001 [0.001, 0.001] | 0→0 | 0→0 | 0.993 |
| symmetric/Float32/n1414/s100/b1/triangle/dropzeros | 255.5 | 241.5 | 0.931 [0.885, 1.056] | 0→0 | 0→0 | 1.002 |
| symmetric/Float32/n1414/s1000/b1/triangle/active_sweep | 3211.0 | 3191.5 | 1.025 [0.984, 1.051] | 0→0 | 0→0 | 1.007 |
| symmetric/Float32/n1414/s1000/b1/triangle/matmat | 13932.0 | 16306.5 | 1.177 [1.142, 1.202] | 0→0 | 0→0 | 1.037 |
| symmetric/Float32/n1414/s10000/b1/triangle/matmat | 103548.5 | 127249.0 | 1.177 [1.156, 1.196] | 0→0 | 0→0 | 0.994 |
| symmetric/Float32/n1414/s10000/b1/triangle/write_active | 24960.0 | 25151.0 | 1.022 [0.996, 1.065] | 0→0 | 0→0 | 0.988 |
| symmetric/Float32/n144/s100/b1/triangle/matmat | 968.0 | 1110.0 | 1.153 [1.144, 1.161] | 0→0 | 0→0 | 0.998 |
| symmetric/Float32/n144/s1000/b1/triangle/matmat | 9312.0 | 10953.5 | 1.172 [1.161, 1.312] | 0→0 | 0→0 | 0.995 |
| symmetric/Float32/n144/s10000/b1/triangle/bilinear | 16363.0 | 32209.0 | 1.969 [1.949, 1.976] | 0→0 | 0→0 | 1.000 |
| symmetric/Float32/n144/s10000/b1/triangle/matmat | 95114.0 | 109371.5 | 1.154 [1.142, 1.165] | 0→0 | 0→0 | 0.997 |
| symmetric/Float64/n1414/s100/b1/triangle/active_sweep | 903.5 | 888.5 | 0.952 [0.908, 1.058] | 0→0 | 0→0 | 1.008 |
| symmetric/Float64/n1414/s100/b1/triangle/zero_active | 1039.5 | 1026.5 | 0.992 [0.940, 1.052] | 0→0 | 0→0 | 1.000 |
| symmetric/Float64/n1414/s1000/b1/triangle/matmat | 20238.5 | 21708.0 | 1.060 [1.048, 1.083] | 0→0 | 0→0 | 0.984 |
| symmetric/Float64/n1414/s1000/b1/triangle/read_diagonal | 3394.5 | 3395.0 | 1.002 [0.985, 1.017] | 0→0 | 0→0 | 1.005 |
| symmetric/Float64/n1414/s1000/b1/triangle/zero_active | 3568.5 | 3537.0 | 0.988 [0.933, 1.017] | 0→0 | 0→0 | 1.002 |
| symmetric/Float64/n1414/s10000/b1/triangle/matmat | 133221.0 | 146131.0 | 1.105 [1.084, 1.116] | 0→0 | 0→0 | 1.010 |
| symmetric/Float64/n1414/s10000/b1/triangle/zero_active | 25775.0 | 25912.0 | 0.990 [0.951, 1.045] | 0→0 | 0→0 | 1.012 |
| symmetric/Float64/n144/s100/b1/triangle/matmat | 1012.5 | 1161.0 | 1.159 [1.127, 1.175] | 0→0 | 0→0 | 1.040 |
| symmetric/Float64/n144/s1000/b1/triangle/matmat | 9300.5 | 11050.0 | 1.186 [1.172, 1.209] | 0→0 | 0→0 | 1.021 |
| symmetric/Float64/n144/s10000/b1/triangle/bilinear | 16405.5 | 32219.0 | 1.955 [1.938, 1.971] | 0→0 | 0→0 | 1.004 |
| symmetric/Float64/n144/s10000/b1/triangle/matmat | 94456.0 | 110785.0 | 1.180 [1.149, 1.397] | 0→0 | 0→0 | 1.013 |
