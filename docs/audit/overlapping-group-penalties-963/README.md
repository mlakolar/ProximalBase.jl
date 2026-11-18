# Probes for the note on nested and tree-structured group penalties (#963)

The scripts behind the numerical statements of
[`docs/design/overlapping-group-penalties-963.md`](../../design/overlapping-group-penalties-963.md).
Each prints one line per recorded statement (`PASS` or `FAIL`) and exits with a nonzero status when one fails.
Recorded instances are literal values in the scripts; random sweeps use seeded `Xoshiro` generators and print their
counts. Sweep and iteration counts (lines C and N) are an illustration, not a performance claim; no timing is
recorded.

## Method

- `harness.jl` (included by the probes, not run alone): group steps for weighted `ℓ2`, `ℓ∞`, `ℓ1` and
  infinite-weight groups, one pass with its dual blocks rebuilt from the steps, cyclic dual block coordinate descent, duality gaps and
  the objective gradient, generic in the element type.
- A candidate is accepted when its duality gap, evaluated in `BigFloat` (256 bits), is at most
  `1e-50 (1 + ‖y‖²)` for a `BigFloat` candidate or `1e-12 (1 + ‖y‖²)` for a `Float64` one. The dual blocks are
  scaled into their dual balls before the gap is formed, so the gap bounds `½‖x − x*‖²`.
- A candidate is refuted against a reference from dual block coordinate descent in `Float64` (gap at most
  `1e-15 (1 + ‖y‖²)`, evaluated again in `BigFloat`): its distance from the reference must exceed
  `2√(2 gap) + 1e-12`. Where all group norms are nonzero, the gradient of the objective is reported as a second
  certificate.

## Scripts and commands

Run from the repository root.

| Script | Shows | Command |
|---|---|---|
| `probe_laminar.jl` | Propositions 1–3: disjoint groups against `prox` of `GroupNormL2` on master, laminar scalar weights, the sparse-group order, `ℓ∞` and mixed norms | `julia --project=. --check-bounds=yes docs/audit/overlapping-group-penalties-963/probe_laminar.jl` |
| `probe_weighted.jl` | Propositions 4–5: diagonal weights inside a group, necessity for two groups, an `ℓ1` group and a zero contained weight (sufficient, not necessary), duplicates, a diagonal step | `julia --project=. --check-bounds=yes docs/audit/overlapping-group-penalties-963/probe_weighted.jl` |
| `probe_banding.jl` | Section 5: reduction to band norms, the general weights of Bien, Bunea and Xiao, the weights of Yu and Bien, two-value weights, the small-step expansion, zero bands, `λ_max`, dual block coordinate descent and a Newton-type method | `julia --project=. --check-bounds=yes docs/audit/overlapping-group-penalties-963/probe_banding.jl` |
| `check_inventory.py` | each `path:line` of the note's inventory holds the named symbol at the pinned commit | `python3 docs/audit/overlapping-group-penalties-963/check_inventory.py` |

## Recorded run

Local, Julia 1.13.4, at commit `2b370ace323a83709d0fbad778c7036534aea53e` (the commit of the scripts
and the note as they stand; this README was updated afterwards). Output lines as printed:

### `probe_laminar.jl`

```text
PASS D1 recorded disjoint instance, master prox(GroupNormL2) = one pass of the harness: max difference 4.44e-16, BigFloat gap of the master point 1.22e-15
PASS D2 200 random disjoint instances (seeded), master = harness: largest difference 1.78e-15, largest gap/(1+‖y‖²) 1.14e-16
PASS L1 recorded laminar instance (forest, duplicate, five nested groups, threshold, 0 and Inf weights), one pass exact: order [2, 7, 8, 1, 9, 3, 4, 5, 6], BigFloat gap 5.53e-76, x[1:2] = (0.0, 0.0), x[11:12] = (9.0, -9.0)
PASS L2 the same instance in another inclusion order gives the same point: BigFloat gap 5.53e-76
PASS L3 300 random laminar families (seeded; forests, duplicates, coordinates in no group, 0 and Inf weights), scalar ℓ2 weights, one pass exact: 285 with a penalized group left nonzero, largest BigFloat gap/(1+‖y‖²) 1.16e-77
PASS W1 sparse-group, coordinates before the group: one pass exact: x = (0.0, 0.0), BigFloat gap 0
PASS W2 sparse-group, the group before the coordinates (wrong order): refuted: x = (0.0024814048950054326, 0.0), distance from the reference 0.00248 > 2√(2 gap) = 0 (reference: 1 sweeps)
PASS M1 300 random laminar families (seeded), ℓ∞ with scalar weights on all groups, one pass exact: 299 with a penalized group left nonzero, largest BigFloat gap/(1+‖y‖²) 7.66e-78
PASS M2 300 random laminar families (seeded), scalar ℓ2 on each containing group, ℓ∞ or diagonal-weight ℓ2 on the others, one pass exact: 298 with a penalized group left nonzero, largest BigFloat gap/(1+‖y‖²) 1.03e-77
PASS M3 an ℓ2 group {1,2} inside an ℓ∞ group {1,2,3}, y = (3, 1, 0): one pass refuted: one pass (1.0513167019494862, 0.6837722339831621, 0.0), reference (1.1055728090000843, 0.5527864045000421, 0.0), distance 0.142 > 2√(2 gap) = 3.85e-08
all statements hold
```

### `probe_weighted.jl`

```text
PASS P1 300 random laminar families (seeded; 277 with nested groups), diagonal weights constant on each contained group, zero weights included, one pass exact: 295 with a penalized group left nonzero, largest BigFloat gap/(1+‖y‖²) 9.78e-78
PASS P2 two groups {1,2} ⊂ {1,2,3}, weights (1,2) and (1,3,2), y = (2,3,1), t = 0.3: one pass refuted: one pass (1.7932769499184134, 1.634182330771852, 0.8201403025867257), reference (1.7636372108070313, 1.660040692786047, 0.8217713839139037), distance 0.0394 > 2√(2 gap) = 6e-08, largest gradient entry at the one-pass point 0.0359 (all group norms nonzero)
PASS P5 an ℓ1 group {1,2} inside ‖diag(1,3,2) z‖₂ on {1,2,3}: one pass exact although the weights differ on {1,2}: y = (3.0, -2.0, 1.0): x = (2.551449046073624, -0.959109715890919, 0.7195992234821781); y = (5.0, 4.0, -2.0): x = (3.773028839174305, 1.6432125472623171, -1.4181883812242777), largest BigFloat gap 0
PASS P3 duplicates g₁ = g₂ = {1,2,3}, diagonal weights (1,2,3) then the scalar weight 1.5: one pass exact: x = (2.3410859616795348, -1.3505441490776786, 2.205965996464773), BigFloat gap 0
PASS P4 the same duplicates, the scalar-weight group first: one pass refuted: distance 0.103 > 2√(2 gap) = 1.14e-07
PASS P7 ‖diag(1,0) z‖₂ on {1,2} inside ‖diag(1,3,2) z‖₂ on {1,2,3}, y = (4, −3, 2), t = 0.4: one pass exact: x = (3.4240704310473182, -2.0513912033125448, 1.659033770482706), BigFloat gap 6.91e-77
PASS P6 duplicates ‖z‖₂ then ‖z‖∞ on {1,2}, y = (3, 1), t = 1: the closed-form point is certified and one pass differs from it: one pass (1.0513167019494862, 0.6837722339831621), proximal point (1.105572809000084, 0.552786404500042) (BigFloat gap 0), distance 0.142
PASS S1 sparse-group penalty with the diagonal step τ = (0.5, 1, 2, 4, 3): one pass exact: x = (3.1098095611881957, -0.25985888054039324, 2.2919211968835707, 0.3090442947109622, -1.0), BigFloat gap 6.91e-77
PASS S2 300 random laminar families (seeded), scalar weights, a diagonal step constant on each contained group, one pass exact: 292 with a penalized group left nonzero, largest BigFloat gap/(1+‖z‖²) 1.46e-77
PASS S3 scalar weights on {1,2} ⊂ {1,2,3} with the step τ = (1, 4, 1), not constant on {1,2}: one pass refuted: distance (in z) 0.0273 > 2√(2 gap) = 1.05e-07
all statements hold
```

### `probe_banding.jl`

```text
PASS B1 p = 5, general weights, λ = 0.25: the band-norm solution, scaled back, has a matrix duality gap at rounding level and keeps the diagonal: band norms (1.2727922061357855, 1.104536101718726, 1.216552506059644, 3.2403703492039297) -> (0.7068846802362903, 0.5518188855137572, 0.6399185778724387, 2.54224888255622), matrix gap 1.78e-16, band-norm gap 3.23e-16, 5 sweeps; one pass differs by 0.0119
PASS B2 the disjoint control (6) as nested groups with zero weights: one pass = prox(GroupNormL2) on master over the bands: largest difference 1.11e-16
PASS B3 recorded instance p = 4, general weights: one pass (Float64) reproduces the posted point: one pass (0.880669128402111, 0.43980506851334233, 0.15773452285496928)
PASS B4 recorded instance: one pass refuted by the certified reference and by the gradient: reference (0.8339820745637797, 0.49606635020397205, 0.15980235543181953) (14 sweeps, BigFloat gap 8.02e-15), distance 0.0731 > 2√(2 gap) = 2.53e-07, largest gradient entry 0.244 at the one-pass point and 1.39e-07 at the reference, objective excess 0.01022428370907346
PASS F basic weights (7), p = 4, 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS F basic weights (7), p = 6, 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS F basic weights (7), p = 10, 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS F disjoint control (6), p = 4, 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS F disjoint control (6), p = 6, 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS F disjoint control (6), p = 10, 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS F general weights (8), p = 3, 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS F general weights (8), p = 4, 100 random instances (seeded): one pass accepted in 19, refuted in 81, undecided in 0; zero-band count differs in 0
PASS F general weights (8), p = 6, 100 random instances (seeded): one pass accepted in 8, refuted in 92, undecided in 0; zero-band count differs in 0
PASS F general weights (8), p = 10, 100 random instances (seeded): one pass accepted in 2, refuted in 98, undecided in 0; zero-band count differs in 0
PASS F Yu-Bien weights 1/(ℓ-m+1)², 2 penalized coordinates (row 3), 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS F Yu-Bien weights 1/(ℓ-m+1)², 3 penalized coordinates (row 4), 100 random instances (seeded): one pass accepted in 24, refuted in 76, undecided in 0; zero-band count differs in 0
PASS F Yu-Bien weights 1/(ℓ-m+1)², 5 penalized coordinates (row 6), 100 random instances (seeded): one pass accepted in 3, refuted in 97, undecided in 0; zero-band count differs in 0
PASS F Yu-Bien weights 1/(ℓ-m+1)², 9 penalized coordinates (row 10), 100 random instances (seeded): one pass accepted in 0, refuted in 100, undecided in 0; zero-band count differs in 0
PASS F two-value weights a_ℓ (m < ℓ), √(2ℓ) (m = ℓ), p = 4, 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS F two-value weights a_ℓ (m < ℓ), √(2ℓ) (m = ℓ), p = 6, 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS F two-value weights a_ℓ (m < ℓ), √(2ℓ) (m = ℓ), p = 10, 100 random instances (seeded): one pass accepted (BigFloat gap) in 100 of 100
PASS Y recorded instance, Yu-Bien weights, three penalized coordinates, y = (2, 1.5, 1), step 0.5: one pass refuted: one pass (1.4440890330680143, 0.980076615736529, 0.5573845210878831), reference (1.4427610008814504, 0.981943801904041, 0.5574710002665637), distance 0.00229 > 2√(2 gap) = 1.05e-07
PASS E one-pass gradient = −t² C(y) + O(t³) at t = 1e-40, C = 0 for K = 2 and C ≠ 0 for K ≥ 3 (general and Yu-Bien weights): general K=2 ‖C‖=0, general K=3 ‖C‖=0.0709, general K=5 ‖C‖=0.236, general K=10 ‖C‖=0.576, general K=30 ‖C‖=1.32, Yu-Bien K=2 ‖C‖=0, Yu-Bien K=3 ‖C‖=0.00308, Yu-Bien K=5 ‖C‖=0.00625, Yu-Bien K=10 ‖C‖=0.00833, Yu-Bien K=30 ‖C‖=0.00914
PASS Z p = 4, λ = 1, band norms (2.05, 3, 0): one pass keeps bands 1 and 2, the proximal point is zero: one pass (0.0014776018749914932, 0.0015847207086445254, 0.0); BCD point at 0.999λ (2.614858704708129e-14, 2.9219029893762624e-14, 0.0); dual norms of the decomposition r = ξ₁ + ξ₂ + ξ₃ (0.999, 0.9989999999999999, 0.9990000000000364) < λ (266 sweeps at 0.999λ)
PASS L p = 3, general weights, band norms (2, 0), λ = 1 below max ‖S_sℓ‖/w_ℓℓ = 1.41: the proximal point is diagonal: x = (0.0, 0.0), BigFloat gap 0
PASS CA p = 8, general weights, λ = 1.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (1, 3.82) (1, 5.43) (1, 7.87) (1, 10.3) (1, 10.5) (gap reached in 5 of 5), nonzero bands (1, 1) of 7, largest gap/(1+‖r‖²) 2.43e-17, one pass off by ‖x₁ − x‖/‖r‖ up to 0, kept 0 against √(2 gap) ≤ 8.36e-08
PASS NA p = 8, λ = 1.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 0, 1, 0, 7.87) (2, 0, 1, 0, 3.82) (2, 0, 1, 0, 10.3) (2, 0, 1, 0, 5.43) (2, 0, 1, 0, 10.5)
PASS CA p = 16, general weights, λ = 1.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (3, 5.3) (3, 7.25) (3, 8.54) (4, 6.8) (4, 8.49) (gap reached in 5 of 5), nonzero bands (3, 3) of 15, largest gap/(1+‖r‖²) 4.92e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00324, kept 0 against √(2 gap) ≤ 2.49e-05
PASS NA p = 16, λ = 1.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 1, 1, 0, 8.49) (2, 1, 1, 0, 5.3) (2, 1, 1, 0, 8.54) (2, 1, 1, 0, 6.8) (2, 1, 1, 0, 7.25)
PASS CA p = 32, general weights, λ = 1.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (4, 5.31) (4, 7.26) (4, 7.51) (4, 9.91) (4, 12) (gap reached in 5 of 5), nonzero bands (7, 7) of 31, largest gap/(1+‖r‖²) 7.55e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00685, kept 0 against √(2 gap) ≤ 6.02e-05
PASS NA p = 32, λ = 1.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 1, 1, 0, 7.51) (2, 1, 1, 0, 7.26) (2, 1, 1, 0, 12) (2, 1, 1, 0, 9.91) (2, 2, 1, 0, 5.31)
PASS CA p = 64, general weights, λ = 1.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (4, 7.4) (4, 8.57) (5, 7.35) (5, 13.7) (5, 15.8) (gap reached in 5 of 5), nonzero bands (15, 15) of 63, largest gap/(1+‖r‖²) 6.66e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00731, kept 0 against √(2 gap) ≤ 0.000142
PASS NA p = 64, λ = 1.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 2, 1, 0, 13.7) (2, 2, 1, 0, 7.35) (2, 2, 1, 0, 7.4) (2, 1, 1, 0, 8.57) (2, 2, 1, 0, 15.8)
PASS CA p = 128, general weights, λ = 1.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (5, 9.77) (5, 12.9) (5, 15.1) (5, 17.8) (5, 20.9) (gap reached in 5 of 5), nonzero bands (31, 31) of 127, largest gap/(1+‖r‖²) 1.36e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00677, kept 0 against √(2 gap) ≤ 0.000124
PASS NA p = 128, λ = 1.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 2, 1, 0, 12.9) (2, 2, 1, 0, 20.9) (2, 2, 1, 0, 9.77) (2, 2, 1, 0, 17.8) (2, 2, 1, 0, 15.1)
PASS CB p = 8, general weights, λ = 1.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (4, 1.17) (4, 1.22) (4, 1.73) (4, 1.83) (5, 1.09) (gap reached in 5 of 5), nonzero bands (7, 7) of 7, largest gap/(1+‖r‖²) 5.82e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00649, kept 0 against √(2 gap) ≤ 2.04e-05
PASS NB p = 8, λ = 1.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 2, 0, 0, 1.09) (2, 1, 0, 0, 1.22) (2, 1, 0, 0, 1.83) (2, 1, 0, 0, 1.17) (2, 1, 0, 0, 1.73)
PASS CB p = 16, general weights, λ = 1.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (5, 1.37) (5, 1.7) (5, 1.99) (5, 2.32) (5, 2.9) (gap reached in 5 of 5), nonzero bands (15, 15) of 15, largest gap/(1+‖r‖²) 1.37e-14, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00653, kept 0 against √(2 gap) ≤ 7.22e-06
PASS NB p = 16, λ = 1.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 2, 0, 0, 1.99) (2, 2, 0, 0, 1.7) (2, 2, 0, 0, 2.32) (2, 2, 0, 0, 2.9) (2, 1, 0, 0, 1.37)
PASS CB p = 32, general weights, λ = 1.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (5, 1.3) (5, 1.33) (5, 1.46) (5, 2.32) (5, 3.03) (gap reached in 5 of 5), nonzero bands (31, 31) of 31, largest gap/(1+‖r‖²) 9.31e-14, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0077, kept 0 against √(2 gap) ≤ 3.76e-05
PASS NB p = 32, λ = 1.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 2, 0, 0, 1.33) (2, 2, 0, 0, 2.32) (2, 2, 0, 0, 1.3) (2, 2, 0, 0, 1.46) (2, 2, 0, 0, 3.03)
PASS CB p = 64, general weights, λ = 1.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (5, 1.12) (5, 1.19) (5, 1.82) (5, 2.22) (5, 2.65) (gap reached in 5 of 5), nonzero bands (63, 63) of 63, largest gap/(1+‖r‖²) 3.13e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00676, kept 0 against √(2 gap) ≤ 0.000143
PASS NB p = 64, λ = 1.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 2, 0, 0, 1.82) (2, 2, 0, 0, 1.12) (2, 2, 0, 0, 1.19) (2, 2, 0, 0, 2.22) (2, 2, 0, 0, 2.65)
PASS CB p = 128, general weights, λ = 1.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (5, 1.94) (5, 2.07) (5, 2.38) (5, 2.46) (5, 3.41) (gap reached in 5 of 5), nonzero bands (127, 127) of 127, largest gap/(1+‖r‖²) 7.17e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00639, kept 0 against √(2 gap) ≤ 0.000417
PASS NB p = 128, λ = 1.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 2, 0, 0, 1.94) (2, 2, 0, 0, 3.41) (2, 2, 0, 0, 2.46) (2, 2, 0, 0, 2.07) (2, 2, 0, 0, 2.38)
PASS CC p = 8, general weights, λ = 2.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (9, 0.00315) (10, 0.451) (18, 0.048) (22, 0.0634) (35, 0.0229) (gap reached in 5 of 5), nonzero bands (7, 7) of 7, largest gap/(1+‖r‖²) 8.9e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0214, kept 0 against √(2 gap) ≤ 2.37e-05
PASS NC p = 8, λ = 2.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 3, 0, 0, 0.0634) (2, 4, 0, 0, 0.0229) (2, 2, 0, 0, 0.00315) (2, 2, 0, 0, 0.451) (2, 3, 0, 0, 0.048)
PASS CC p = 16, general weights, λ = 2.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (11, 0.0299) (12, 0.0284) (13, 0.0167) (24, 0.00112) (30, 0.077) (gap reached in 5 of 5), nonzero bands (15, 15) of 15, largest gap/(1+‖r‖²) 9.91e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0316, kept 0 against √(2 gap) ≤ 6.16e-05
PASS NC p = 16, λ = 2.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 3, 0, 0, 0.0284) (2, 4, 0, 0, 0.00112) (2, 4, 0, 0, 0.077) (2, 3, 0, 0, 0.0167) (2, 3, 0, 0, 0.0299)
PASS CC p = 32, general weights, λ = 2.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (27, 0.0827) (30, 0.019) (45, 0.00465) (90, 0.00965) (356, 0.00407) (gap reached in 5 of 5), nonzero bands (31, 31) of 31, largest gap/(1+‖r‖²) 9.84e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0334, kept 0 against √(2 gap) ≤ 0.000134
PASS NC p = 32, λ = 2.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 9, 0, 0, 0.00965) (2, 4, 0, 0, 0.00465) (2, 4, 0, 0, 0.019) (2, 4, 0, 0, 0.0827) (20, 22, 0, 1, 0.00407)
PASS CC p = 64, general weights, λ = 2.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (20, 0.00806) (21, 0.00514) (29, 0.00696) (39, 0.062) (40, 0.0515) (gap reached in 5 of 5), nonzero bands (63, 63) of 63, largest gap/(1+‖r‖²) 8.19e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0376, kept 0 against √(2 gap) ≤ 0.000232
PASS NC p = 64, λ = 2.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 5, 0, 0, 0.0515) (2, 4, 0, 0, 0.00806) (2, 4, 0, 0, 0.00514) (2, 4, 0, 0, 0.00696) (2, 5, 0, 0, 0.062)
PASS CC p = 128, general weights, λ = 2.0, 5 instances (seeded): dual BCD: (sweeps, smallest group) (16, 0.0704) (23, 0.282) (24, 0.0757) (30, 0.0434) (33, 0.0468) (gap reached in 5 of 5), nonzero bands (127, 127) of 127, largest gap/(1+‖r‖²) 9.61e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0391, kept 0 against √(2 gap) ≤ 0.000491
PASS NC p = 128, λ = 2.0: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 4, 0, 0, 0.0468) (2, 4, 0, 0, 0.0757) (2, 3, 0, 0, 0.0704) (2, 4, 0, 0, 0.282) (2, 4, 0, 0, 0.0434)
PASS CD p = 8, general weights, λ = 2.5, 5 instances (seeded): dual BCD: (sweeps, smallest group) (1, 0.498) (5, 0.456) (19, 0.0802) (43, 0.0128) (60, 0.0194) (gap reached in 5 of 5), nonzero bands (1, 7) of 7, largest gap/(1+‖r‖²) 8.19e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0122, kept 0 against √(2 gap) ≤ 2.78e-05
PASS ND p = 8, λ = 2.5: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 6, 1, 0, 0.0194) (2, 4, 0, 0, 0.0128) (2, 0, 1, 0, 0.498) (2, 3, 1, 0, 0.0802) (2, 2, 1, 0, 0.456)
PASS CD p = 16, general weights, λ = 2.5, 5 instances (seeded): dual BCD: (sweeps, smallest group) (1, NaN) (35, 0.0586) (114, 0.0333) (975, 6.45e-05) (2332, 3.86e-11) (gap reached in 5 of 5), nonzero bands (0, 15) of 15, largest gap/(1+‖r‖²) 1e-12, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0166, kept 0 against √(2 gap) ≤ 6.48e-05
PASS ND p = 16, λ = 2.5: Newton-type method: certified in 4 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 4, 1, 0, 0.0586) (2, 7, 1, 0, 0.0333) (2000, 59, 73, 5, 3.86e-11) (2, 0, 1, 0, NaN) (20, 34, 1, 1, 6.45e-05)
PASS CD p = 32, general weights, λ = 2.5, 5 instances (seeded): dual BCD: (sweeps, smallest group) (1, 2.13) (1, NaN) (10, 0.00903) (2687, 0.00113) (3955, 6.67e-05) (gap reached in 5 of 5), nonzero bands (0, 26) of 31, largest gap/(1+‖r‖²) 1e-12, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0242, kept 0 against √(2 gap) ≤ 0.000142
PASS ND p = 32, λ = 2.5: Newton-type method: certified in 5 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2, 2, 1, 0, 0.00903) (2, 0, 1, 0, NaN) (20, 31, 1, 1, 0.00113) (2, 0, 1, 0, 2.13) (2, 6, 1, 0, 6.67e-05)
PASS CD p = 64, general weights, λ = 2.5, 5 instances (seeded): dual BCD: (sweeps, smallest group) (13, 1.54) (20, 0.714) (328, 2.66e-14) (2760, 0.00255) (100000, 5.6e-10) (gap reached in 4 of 5), nonzero bands (5, 63) of 63, largest gap/(1+‖r‖²) 2.2e-10, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0295, kept 0 against √(2 gap) ≤ 0.00387
PASS ND p = 64, λ = 2.5: Newton-type method: certified in 3 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2000, 47, 13, 5, 2.66e-14) (2, 3, 1, 0, 0.714) (2, 3, 1, 0, 1.54) (20, 14, 2, 1, 0.00255) (2000, 171, 0, 5, 5.6e-10)
PASS CD p = 128, general weights, λ = 2.5, 5 instances (seeded): dual BCD: (sweeps, smallest group) (9, 1.42e-12) (17, 9.78e-13) (620, 4.84e-13) (1358, 2.76e-15) (7591, 6.07e-12) (gap reached in 5 of 5), nonzero bands (97, 126) of 127, largest gap/(1+‖r‖²) 1e-12, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0308, kept 0 against √(2 gap) ≤ 0.000529
PASS ND p = 128, λ = 2.5: Newton-type method: certified in 0 of 5 (0 by the tolerance rule); (proposal sweeps, Newton iterations, prefix-certificate sweeps, failed attempts, smallest group) (2000, 86, 0, 5, 6.07e-12) (2000, 19, 9, 5, 1.42e-12) (2000, 67, 21, 5, 4.84e-13) (2000, 77, 91, 5, 2.76e-15) (2000, 17, 7, 5, 9.78e-13)
all statements hold
```

### `check_inventory.py`

```text
87 citations, 0 failure(s)
```

## Literature addendum probe

`probe_literature.jl` checks the numerical statements of
[`docs/design/overlapping-group-penalties-963-literature.md`](../../design/overlapping-group-penalties-963-literature.md):
weighted `ℓ∞` groups with proportional weights and the exact counterexample (L1, the counterexample in rational
arithmetic and refuted by a directional derivative), a radial function on a maximal group and on a contained group
(L2; the certificate evaluates the exact indicator of a ball after moving a candidate that lies outside it by rounding
into it, and adds the distance moved), weighted `ℓ1` groups inside `ℓ∞` groups and a duplicate-group counterexample in rational arithmetic (L3), the weights (W3) of Jenatton, Audibert and Bach on a chain, the
latent penalty on a nested chain of Yan and Bien (L4: Algorithm 3 as published certified by a `BigFloat` duality gap
from an explicit decomposition, the redundancy rule, ties, the hull form and the value formula), `λ_max` of the
banding prox (L5), and dual block coordinate descent with every group at its threshold (line O, an observation). It
includes `harness.jl` and adds its own group functions; acceptance and refutation follow the Method section above,
with the exceptions named in the lines.

Command, from the repository root:
`julia --project=. --check-bounds=yes docs/audit/overlapping-group-penalties-963/probe_literature.jl`

### Recorded run of `probe_literature.jl`

Local, Julia 1.13.4, at commit `d6af986a765e9e77a3c3c337a549d295bbe2b048` (the commit of the script and the addendum
as they stand; this section was added afterwards). The three probes above, run again at that commit, printed the lines
recorded above. Output lines as printed:

```text
PASS L1 300 random laminar families (seeded), η‖D z‖∞ with D_j on g_k proportional to D_k, η = 0 and ∞ included, one pass exact: 275 with a group moved and not zero, largest BigFloat gap/(1+‖y‖²) 1.29e-77
PASS L1 300 random laminar families (seeded), one weight vector D shared by all groups, one pass exact: 284 with a group moved and not zero, largest BigFloat gap/(1+‖y‖²) 1.39e-77
PASS L1 ‖z‖∞ on {1,2} inside ‖diag(1,2,1) z‖∞ on {1,2,3}, η = t = 1, y = (3, 2.9, 0): one pass refuted (exact): one pass 107//50, 107//100, 0//1, directional derivative along −e₁ -7//50
PASS L2 300 random laminar families (seeded), h(‖·‖₂) with h(r) = ωr + μr²/2 (and a ball) on each maximal group, Proposition 2 below it, one pass exact: 292 with a group moved and not zero, largest BigFloat gap/(1+‖y‖²) 1.07e-77
PASS L2 0.8‖z‖₂ on {1,2}, 0.5r + r² (r = ‖z‖₂) on {1,2,3}, y = (2, 3, 1.5), t = 0.3: one pass exact: x = (1.1192959196648145, 1.6789438794972218, 0.8993353169636529), BigFloat gap 0
PASS L2 the radial function on the contained group {1,2} and 0.8‖z‖₂ on {1,2,3}: one pass refuted by the gradient: one pass (1.0886540969427958, 1.6329811454141936, 1.363093013485373), largest gradient entry 0.0984 (all group norms nonzero)
PASS L3 300 random laminar families (seeded), ω‖·‖∞ on each group that contains another (an equal group placed before it included), weighted ℓ1 or ℓ∞ on the others, one pass exact: 294 with a group moved and not zero, largest BigFloat gap/(1+‖y‖²) 4.47e-78
PASS L3 duplicates {1,2}: ‖z‖∞ then |z₁|, t = 1, y = (2, 19/10): one pass refuted (exact), the order with |z₁| first certified: one pass 9//20, 29//20, proximal point 19//20, 19//20 (rational gap 0//1)
PASS W3 chain {1} ⊂ {1,2} ⊂ {1,2,3}, weights ρ^(ℓ−j) with ρ = 0.5, y = (2, 1.5, 1), t = 0.5: one pass refuted: one pass (1.3178492726525024, 0.9398674314954554, 0.6302987403482098), reference (1.3101471285091393, 0.9495843853938258, 0.6309642768020816) (4 sweeps), distance 0.0124 > 2√(2 gap) = 1.98e-07, largest gradient entry 0.0129
PASS L4 Algorithm 3 as published, 300 random nested chains (seeded), strictly increasing weights: certified: 679 knots in all, largest BigFloat gap/(1+‖y‖²) 3.82e-77, every ‖u_{g_ℓ}‖ ≤ λ w_ℓ
PASS L4 300 random chains (seeded) with equal or decreasing weights (202 reduced): groups with w_k ≥ w_j for some j > k removed, Algorithm 3 on the rest, certified against all groups: largest BigFloat gap/(1+‖y‖²) 1.56e-77
PASS L4 the hull variant (each segment y (1 − λ/√slope)₊) equals Algorithm 3 on the 600 chains: largest difference 8.64e-77
PASS L4 the value formula Σ √(Δw²)‖y_seg‖ on the hull lies between a dual lower bound and a decomposition upper bound on the 600 chains: largest (upper − lower)/(1 + value) 3.91e-77
PASS L4 exact ties in line 5 (y = (1 | 1, 1, 1 | 1, 2), w = (1, 2, 3), λ = 1/2): first and last maximizer both certified, same point: knots [1, 2, 3] and [3], x = y/2, BigFloat gaps 0 and 0
PASS L5 p = 3, general weights, band norms (2, 0): λ_max = Ω°(r) between equal bounds, below max ‖S_sℓ‖/w_ℓℓ = 1.41: lower 0.828, upper 0.828 (2√2 − 2 = 0.828)
PASS O p = 4, general weights, every group at its threshold (c = 1) and inside it (c = 0.999): dual BCD distance after 10, 100, 1000, 10000 sweeps: c = 1: 0.00914, 0.00156, 0.000168, 1.7e-05; c = 0.999: 0.00694, 0.000373, 1.15e-12, 7.41e-16
all statements hold
```
