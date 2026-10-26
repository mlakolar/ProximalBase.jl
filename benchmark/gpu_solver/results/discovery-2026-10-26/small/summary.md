# Common-accuracy solver timings

Times include stopping checks. Failed samples are excluded from speedups and counted below.
Variability is the range of repetition times; seed-specific medians remain separate.
CPU/GPU is a ratio of medians, not a claim of a universal crossover.

| n × p | type | seed | scenario | tolerance | check | CPU median [min,max] ms | GPU median [min,max] ms | CPU/GPU | failures / total |
|---|---|---|---|---|---|---|---|---|---|
| 400 × 40 | Float32 | 103 | resident | 0.0001 | 1 | 0.026 [0.022,0.057] | 2.103 [2.085,2.207] | 0.012 | 0/10 |
| 400 × 40 | Float32 | 103 | transfers | 0.0001 | 1 | 0.030 [0.024,0.037] | 2.136 [2.019,2.164] | 0.014 | 0/10 |
| 400 × 40 | Float32 | 103 | resident | 0.0001 | 10 | 0.037 [0.032,0.047] | 2.270 [2.219,2.381] | 0.016 | 0/10 |
| 400 × 40 | Float32 | 103 | transfers | 0.0001 | 10 | 0.038 [0.035,0.050] | 2.291 [2.195,2.337] | 0.017 | 0/10 |
| 400 × 40 | Float64 | 103 | resident | 0.0001 | 1 | 0.061 [0.045,0.082] | 2.153 [2.053,2.246] | 0.028 | 0/10 |
| 400 × 40 | Float64 | 103 | transfers | 0.0001 | 1 | 0.050 [0.047,0.062] | 2.212 [2.124,2.256] | 0.023 | 0/10 |
| 400 × 40 | Float64 | 103 | resident | 0.0001 | 10 | 0.068 [0.065,0.077] | 2.181 [2.100,2.186] | 0.031 | 0/10 |
| 400 × 40 | Float64 | 103 | transfers | 0.0001 | 10 | 0.070 [0.067,0.080] | 2.272 [2.201,2.316] | 0.031 | 0/10 |

Recorded samples: 80. Failed samples: 0.

Setup and first-call costs are recorded separately in setup-*.toml. First-call time includes compilation and execution; it is not pure compilation time.
The transfer scenario includes workspace construction, upload, solve, and final coefficient download. Resident verification downloads are outside the clock.
See environment.txt and the saved manifest for hardware and versions. Profiling and explicit-GC diagnostics are separate runs.
