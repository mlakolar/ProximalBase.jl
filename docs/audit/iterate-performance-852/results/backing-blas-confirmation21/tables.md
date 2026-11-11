# Measured primary comparisons

R = post850/original; intervals resample paired process rounds. Values are ns per complete case.

| Case | Original ns | Post850 ns | R [95% CI] | Bytes old→new | Allocs old→new | A/A |
|---|---:|---:|---|---|---|---:|
| block/Float32/n8192/s102/b8/arbitrary/dot_backing_native | 478.0 | 766.0 | 1.596 [1.316, 1.630] | 0→0 | 0→0 | 1.002 |
| block/Float32/n8192/s102/b8/strided/dot_backing_native | 467.0 | 555.5 | 1.190 [1.181, 1.199] | 0→0 | 0→0 | 1.011 |
| block/Float32/n8192/s12/b64/strided/dot_backing_native | 471.0 | 502.5 | 0.998 [0.968, 1.047] | 0→0 | 0→0 | 1.002 |
| block/Float32/n8192/s819/b1/arbitrary/dot_backing_native | 479.0 | 1143.0 | 2.380 [1.930, 2.455] | 0→0 | 0→0 | 1.018 |
| block/Float32/n8192/s819/b1/contiguous/dot_backing_native | 593.0 | 923.0 | 1.559 [1.540, 1.566] | 0→0 | 0→0 | 1.003 |
| block/Float32/n8192/s819/b1/strided/dot_backing_native | 475.0 | 979.5 | 2.059 [1.670, 2.099] | 0→0 | 0→0 | 1.021 |
| block/Float64/n65536/s819/b8/arbitrary/dot_backing_native | 5465.0 | 11194.5 | 2.065 [2.017, 2.122] | 0→0 | 0→0 | 0.971 |
| block/Float64/n8192/s102/b8/arbitrary/dot_backing_native | 776.0 | 832.5 | 1.076 [1.071, 1.079] | 0→0 | 0→0 | 0.999 |
| block/Float64/n8192/s819/b1/arbitrary/dot_backing_native | 777.0 | 1196.5 | 1.536 [1.515, 1.550] | 0→0 | 0→0 | 0.997 |
| block/Float64/n8192/s819/b1/contiguous/dot_backing_native | 784.0 | 969.0 | 1.230 [0.993, 1.246] | 0→0 | 0→0 | 1.001 |
| block/Float64/n8192/s819/b1/strided/dot_backing_native | 780.0 | 1022.5 | 1.301 [1.295, 1.320] | 0→0 | 0→0 | 1.002 |
