Validated 1080 unique samples and numerical success claims (full).

Times to residual 1e-4, checked every update. Each range spans the three seed medians; each median uses five repetitions. CPU/GPU ratios are paired within seed.

| Shape | Type | Scenario | CPU ms | GPU ms | CPU/GPU |
|---|---|---|---|---|---|
| 400 × 40 | Float32 | resident | 0.026–0.047 | 1.964–2.886 | 0.012–0.016 |
| 400 × 40 | Float32 | transfers | 0.027–0.048 | 2.044–2.954 | 0.013–0.016 |
| 400 × 40 | Float64 | resident | 0.046–0.085 | 2.004–2.921 | 0.022–0.029 |
| 400 × 40 | Float64 | transfers | 0.047–0.086 | 2.042–3.078 | 0.022–0.028 |
| 4000 × 400 | Float32 | resident | 1.587–1.688 | 2.649–2.797 | 0.597–0.612 |
| 4000 × 400 | Float32 | transfers | 1.578–1.674 | 3.916–4.053 | 0.398–0.413 |
| 4000 × 400 | Float64 | resident | 3.571–3.853 | 2.693–2.879 | 1.326–1.396 |
| 4000 × 400 | Float64 | transfers | 3.458–3.702 | 5.337–5.419 | 0.647–0.683 |
| 10000 × 1000 | Float32 | resident | 20.087–21.238 | 2.987–3.216 | 6.574–6.725 |
| 10000 × 1000 | Float32 | transfers | 20.061–21.236 | 11.443–11.698 | 1.742–1.815 |
| 10000 × 1000 | Float64 | resident | 47.685–51.496 | 4.511–4.804 | 10.526–10.719 |
| 10000 × 1000 | Float64 | transfers | 47.998–51.476 | 21.294–21.548 | 2.253–2.389 |

Maximum independent residuals:
0.0001 9.910244012783133e-05
1.0e-6 9.5930032362706e-07
Maximum absolute objective gap: 2.6219110793590517e-07

Original diagnostic runs (separate from benchmark samples):

| Shape | Check | Instrument/GC policy | Median ms | Min–max ms |
|---|---|---|---|---|
| 400 | 1 | wall_natural_gc | 2.686 | 2.656–2.966 |
| 400 | 1 | julia_timed | 2.658 | 2.638–2.841 |
| 400 | 1 | cuda_timed | 2.656 | 2.637–2.824 |
| 400 | 1 | wall_explicit_gc | 2.950 | 2.913–3.133 |
| 400 | 10 | wall_natural_gc | 2.972 | 2.963–3.092 |
| 400 | 10 | julia_timed | 2.957 | 2.940–3.040 |
| 400 | 10 | cuda_timed | 2.960 | 2.945–3.044 |
| 400 | 10 | wall_explicit_gc | 3.234 | 3.207–3.300 |
| 10000 | 1 | wall_natural_gc | 4.539 | 4.491–4.692 |
| 10000 | 1 | julia_timed | 4.503 | 4.491–4.604 |
| 10000 | 1 | cuda_timed | 4.505 | 4.488–4.573 |
| 10000 | 1 | wall_explicit_gc | 4.707 | 4.690–4.794 |
| 10000 | 10 | wall_natural_gc | 5.095 | 5.085–5.432 |
| 10000 | 10 | julia_timed | 5.084 | 5.078–5.100 |
| 10000 | 10 | cuda_timed | 5.086 | 5.072–5.426 |
| 10000 | 10 | wall_explicit_gc | 5.285 | 5.252–5.328 |
