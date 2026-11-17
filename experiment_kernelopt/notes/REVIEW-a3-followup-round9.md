**Round 9 — recommendation: C, universal, after repair and measurement.** N’s projection holds approximately in aggregate, but its measured NormL1 losses make it a less attractive compromise. I now prefer a static byte cutoff just above 16 KiB, retaining the `y === x` exemption; the maintainer must accept its lost good-placement gains, short-call costs and extra code, or consciously choose U’s broader gains with material ordinary-allocation regressions.

**Verification.** Independently reread both matrices: all 36 processes pass round-8 validity checks, with 1008 cells each and matching recorded tree commits. Ratios below retain round 8’s **median of 21 paired-round ratios**, then median across six processes for equal-cell geometric means; times are separate medians. Both new Zen jobs used a02-03, one additional matrix node. Scripts, raw-input hashes, validation and full process tables are in [round9](round9/README.md); no cluster contact or writes outside review-code. Existing source/API conclusions remain trusted.

**1. Approximately right; the control claim is false literally.** Across 504 ElasticNet/weighted-ElasticNet cells, N/M4 geometric means are **0.999565 Broadwell, 0.999409 Skylake-SP, 0.998752 Zen**, with control means **0.999775, 1.000135, 0.995756**. Individual Zen exceptions exceed control variation: weighted ElasticNet `prox_incopy/Float64/n4`, b1, is **1.546347**, control **0.984584**, **17.7619→27.4540 ns**; unweighted `prox_out/Float64/n32768`, b3, is **1.398888**, control **1.226747**, **7.8733→11.1144 µs**. Neither establishes a replicated N-specific regression; neither permits “the control deviates as much.” [All process ratios, controls and times](round9/evidence.md).

**2. Real losses, but the quoted U ranges are too strong.** For Zen `l1/prox_out/Float32`, a2 means job a/process 2:

| n/process | N/M4 | U/M4 | A/A | M4→N / U, µs |
|:--|--:|--:|--:|:--|
| 32768/a2 | 1.247932 | 1.242381 | 1.004916 | 3.4326→4.2886 / 4.2803 |
| 32768/a3 | 1.226273 | 1.143055 | 0.991444 | 3.4809→4.2850 / 4.2841 |
| 65536/b1 | 1.171634 | 1.170938 | 1.003728 | 7.4340→8.7043 / 8.7052 |
| 65536/b2 | 1.225480 | 1.134892 | 0.988589 | 7.0796→8.6752 / 8.6565 |
| 65536/b3 | 1.213883 | 1.107113 | 1.006185 | 7.1179→8.6493 / 7.9021 |

Ratios of separate medians explain some differences (32768/a3 gives U≈1.231), but do not recover your U ranges throughout. Other cells’ controls span **0.814487–1.291036**, but some large losses remain clean: N’s Float32 n1048576/a1 is **1.268279**, control **1.002232**, **110.4038→139.8781 µs**. The new a04-08 allocation probe independently has **26 clean-control unadvised NormL1 losses >1.05**; worst **1.222259**, control **1.001288**, **14.2141→17.3733 µs**. Thus residual exposure is well supported, without attributing differences causally to the node.

**3. Your aggregate numbers reproduce.** Ranges span matrices 3/4, pooling each campaign’s six Zen processes:

| CPU | U | Zen-only 32 KiB | universal 32 KiB | 2 MiB | N |
|:--|--:|--:|--:|--:|--:|
| Broadwell | .850523–.850900 | .856876–.857594 | .865489 | .855103 | .913315 |
| Skylake-SP | .917183–.923785 | .921507–.929452 | .930455 | .921881 | .954851 |
| Zen 3 | .882879–.890504 | .900076–.900227 | .898353 | .890846 | .938172 |

**4. Directionally right; “dominated” overclaims.** N loses 0.031–0.063 against U’s mean, but remains simpler; C sacrifices genuine wins and adds tiny-call regressions. The repaired lower cutoff is unmeasured, so these descriptive means cannot establish its superiority or workload frequencies.

**Before C merges:** prefer universal coverage because Intel placement losses also exist, and a static rule avoids CPU detection/initialization/Ref machinery. Repair zero-size element types, define boundary rounding precisely, and measure the actual candidate: both types around 16–20–32 KiB, controlled offsets/page shifts and verified backing, ordinary allocations across Zen nodes, and all-three-CPU matrices including tiny calls and nonzero in-place inputs. Repeat API/differential checks at the new boundary; demonstrate rescue and quantify sacrificed gains. This is a candidate recommendation, not approval of the existing 32 KiB trees.
