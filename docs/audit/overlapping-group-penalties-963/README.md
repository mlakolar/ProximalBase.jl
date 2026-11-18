# Probes for the note on nested and tree-structured group penalties (#963)

The scripts behind the numerical statements of
[`docs/design/overlapping-group-penalties-963.md`](../../design/overlapping-group-penalties-963.md).
Each prints one line per recorded statement (`PASS` or `FAIL`) and exits with a nonzero status when one fails.
Recorded instances are literal values in the scripts; random sweeps use seeded `Xoshiro` generators and print their
counts. Sweep counts of the reference solver are an illustration, not a performance claim; no timing is recorded.

## Method

- `harness.jl` (included by the probes, not run alone): group steps for weighted `ℓ2`, `ℓ∞` and infinite-weight
  groups, one pass with its dual blocks rebuilt from the steps, cyclic dual block coordinate descent, duality gaps and
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
| `probe_weighted.jl` | Propositions 4–5: diagonal weights inside a group, necessity for two groups, duplicates, a diagonal step | `julia --project=. --check-bounds=yes docs/audit/overlapping-group-penalties-963/probe_weighted.jl` |
| `probe_banding.jl` | Section 5: reduction to band norms, the general weights of Bien, Bunea and Xiao, the weights of Yu and Bien, two-value weights, `λ_max`, sweeps of dual block coordinate descent | `julia --project=. --check-bounds=yes docs/audit/overlapping-group-penalties-963/probe_banding.jl` |
| `check_inventory.py` | each `path:line` of the note's inventory holds the named symbol at the pinned commit | `python3 docs/audit/overlapping-group-penalties-963/check_inventory.py` |

## Recorded run

Local, Julia 1.13.4, at commit `d48336f5905ff595763c77c4fc82063bacd19bad` (the commit that adds the scripts
and the note in their current form; this README was updated afterwards). Output lines as printed:

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
PASS P3 duplicates g₁ = g₂ = {1,2,3}, diagonal weights (1,2,3) then the scalar weight 1.5: one pass exact: x = (2.3410859616795348, -1.3505441490776786, 2.205965996464773), BigFloat gap 0
PASS P4 the same duplicates, the scalar-weight group first: one pass refuted: distance 0.103 > 2√(2 gap) = 1.14e-07
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
PASS F basic weights (7), p = 4, 100 random instances (seeded): one pass exact in 100 of 100
PASS F basic weights (7), p = 6, 100 random instances (seeded): one pass exact in 100 of 100
PASS F basic weights (7), p = 10, 100 random instances (seeded): one pass exact in 100 of 100
PASS F disjoint control (6), p = 4, 100 random instances (seeded): one pass exact in 100 of 100
PASS F disjoint control (6), p = 6, 100 random instances (seeded): one pass exact in 100 of 100
PASS F disjoint control (6), p = 10, 100 random instances (seeded): one pass exact in 100 of 100
PASS F general weights (8), p = 3, 100 random instances (seeded): one pass exact in 100 of 100
PASS F general weights (8), p = 4, 100 random instances (seeded): one pass exact in 19, refuted in 81, undecided in 0; zero-band count differs in 0
PASS F general weights (8), p = 6, 100 random instances (seeded): one pass exact in 8, refuted in 92, undecided in 0; zero-band count differs in 0
PASS F general weights (8), p = 10, 100 random instances (seeded): one pass exact in 2, refuted in 98, undecided in 0; zero-band count differs in 0
PASS F Yu-Bien weights 1/(ℓ-m+1)², p = 3, 100 random instances (seeded): one pass exact in 100 of 100
PASS F Yu-Bien weights 1/(ℓ-m+1)², p = 4, 100 random instances (seeded): one pass exact in 24, refuted in 76, undecided in 0; zero-band count differs in 0
PASS F Yu-Bien weights 1/(ℓ-m+1)², p = 6, 100 random instances (seeded): one pass exact in 3, refuted in 97, undecided in 0; zero-band count differs in 0
PASS F Yu-Bien weights 1/(ℓ-m+1)², p = 10, 100 random instances (seeded): one pass exact in 0, refuted in 100, undecided in 0; zero-band count differs in 0
PASS F two-value weights a_ℓ (m < ℓ), √(2ℓ) (m = ℓ), p = 4, 100 random instances (seeded): one pass exact in 100 of 100
PASS F two-value weights a_ℓ (m < ℓ), √(2ℓ) (m = ℓ), p = 6, 100 random instances (seeded): one pass exact in 100 of 100
PASS F two-value weights a_ℓ (m < ℓ), √(2ℓ) (m = ℓ), p = 10, 100 random instances (seeded): one pass exact in 100 of 100
PASS Y recorded instance, Yu-Bien weights, three coordinates, y = (2, 1.5, 1), step 0.5: one pass refuted: one pass (1.4440890330680143, 0.980076615736529, 0.5573845210878831), reference (1.4427610008814504, 0.981943801904041, 0.5574710002665637), distance 0.00229 > 2√(2 gap) = 1.05e-07
PASS L p = 3, general weights, band norms (2, 0), λ = 1 below max ‖S_sℓ‖/w_ℓℓ = 1.41: the proximal point is diagonal: x = (0.0, 0.0), BigFloat gap 0
PASS CA p = 8, general weights, λ = 1.0, 5 instances (seeded): sweeps of dual BCD: sweeps [1, 1, 1, 1, 1] (gap reached in 5 of 5), nonzero bands (1, 1) of 7, largest factor spread 0, largest gap/(1+‖r‖²) 2.43e-17, one pass off by ‖x₁ − x‖/‖r‖ up to 0, kept 0 against √(2 gap) ≤ 8.36e-08
PASS CA p = 16, general weights, λ = 1.0, 5 instances (seeded): sweeps of dual BCD: sweeps [3, 3, 3, 4, 4] (gap reached in 5 of 5), nonzero bands (3, 3) of 15, largest factor spread 0.0697, largest gap/(1+‖r‖²) 4.92e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00324, kept 0 against √(2 gap) ≤ 2.49e-05
PASS CA p = 32, general weights, λ = 1.0, 5 instances (seeded): sweeps of dual BCD: sweeps [4, 4, 4, 4, 4] (gap reached in 5 of 5), nonzero bands (7, 7) of 31, largest factor spread 0.133, largest gap/(1+‖r‖²) 7.55e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00685, kept 0 against √(2 gap) ≤ 6.02e-05
PASS CA p = 64, general weights, λ = 1.0, 5 instances (seeded): sweeps of dual BCD: sweeps [4, 4, 5, 5, 5] (gap reached in 5 of 5), nonzero bands (15, 15) of 63, largest factor spread 0.136, largest gap/(1+‖r‖²) 6.66e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00731, kept 0 against √(2 gap) ≤ 0.000142
PASS CA p = 128, general weights, λ = 1.0, 5 instances (seeded): sweeps of dual BCD: sweeps [5, 5, 5, 5, 5] (gap reached in 5 of 5), nonzero bands (31, 31) of 127, largest factor spread 0.142, largest gap/(1+‖r‖²) 1.36e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00677, kept 0 against √(2 gap) ≤ 0.000124
PASS CB p = 8, general weights, λ = 1.0, 5 instances (seeded): sweeps of dual BCD: sweeps [4, 4, 4, 4, 5] (gap reached in 5 of 5), nonzero bands (7, 7) of 7, largest factor spread 0.139, largest gap/(1+‖r‖²) 5.82e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00649, kept 0 against √(2 gap) ≤ 2.04e-05
PASS CB p = 16, general weights, λ = 1.0, 5 instances (seeded): sweeps of dual BCD: sweeps [5, 5, 5, 5, 5] (gap reached in 5 of 5), nonzero bands (15, 15) of 15, largest factor spread 0.151, largest gap/(1+‖r‖²) 1.37e-14, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00653, kept 0 against √(2 gap) ≤ 7.22e-06
PASS CB p = 32, general weights, λ = 1.0, 5 instances (seeded): sweeps of dual BCD: sweeps [5, 5, 5, 5, 5] (gap reached in 5 of 5), nonzero bands (31, 31) of 31, largest factor spread 0.165, largest gap/(1+‖r‖²) 9.31e-14, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0077, kept 0 against √(2 gap) ≤ 3.76e-05
PASS CB p = 64, general weights, λ = 1.0, 5 instances (seeded): sweeps of dual BCD: sweeps [5, 5, 5, 5, 5] (gap reached in 5 of 5), nonzero bands (63, 63) of 63, largest factor spread 0.172, largest gap/(1+‖r‖²) 3.13e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00676, kept 0 against √(2 gap) ≤ 0.000143
PASS CB p = 128, general weights, λ = 1.0, 5 instances (seeded): sweeps of dual BCD: sweeps [5, 5, 5, 5, 5] (gap reached in 5 of 5), nonzero bands (127, 127) of 127, largest factor spread 0.171, largest gap/(1+‖r‖²) 7.17e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.00639, kept 0 against √(2 gap) ≤ 0.000417
PASS CC p = 8, general weights, λ = 2.0, 5 instances (seeded): sweeps of dual BCD: sweeps [9, 10, 18, 22, 35] (gap reached in 5 of 5), nonzero bands (7, 7) of 7, largest factor spread 0.524, largest gap/(1+‖r‖²) 8.9e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0214, kept 0 against √(2 gap) ≤ 2.37e-05
PASS CC p = 16, general weights, λ = 2.0, 5 instances (seeded): sweeps of dual BCD: sweeps [11, 12, 13, 24, 30] (gap reached in 5 of 5), nonzero bands (15, 15) of 15, largest factor spread 0.642, largest gap/(1+‖r‖²) 9.91e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0316, kept 0 against √(2 gap) ≤ 6.16e-05
PASS CC p = 32, general weights, λ = 2.0, 5 instances (seeded): sweeps of dual BCD: sweeps [27, 30, 45, 90, 356] (gap reached in 5 of 5), nonzero bands (31, 31) of 31, largest factor spread 0.742, largest gap/(1+‖r‖²) 9.84e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0334, kept 0 against √(2 gap) ≤ 0.000134
PASS CC p = 64, general weights, λ = 2.0, 5 instances (seeded): sweeps of dual BCD: sweeps [20, 21, 29, 39, 40] (gap reached in 5 of 5), nonzero bands (63, 63) of 63, largest factor spread 0.801, largest gap/(1+‖r‖²) 8.19e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0376, kept 0 against √(2 gap) ≤ 0.000232
PASS CC p = 128, general weights, λ = 2.0, 5 instances (seeded): sweeps of dual BCD: sweeps [16, 23, 24, 30, 33] (gap reached in 5 of 5), nonzero bands (127, 127) of 127, largest factor spread 0.81, largest gap/(1+‖r‖²) 9.61e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0391, kept 0 against √(2 gap) ≤ 0.000491
PASS CD p = 8, general weights, λ = 2.5, 5 instances (seeded): sweeps of dual BCD: sweeps [1, 5, 19, 43, 60] (gap reached in 5 of 5), nonzero bands (1, 7) of 7, largest factor spread 0.492, largest gap/(1+‖r‖²) 8.19e-13, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0122, kept 0 against √(2 gap) ≤ 2.78e-05
PASS CD p = 16, general weights, λ = 2.5, 5 instances (seeded): sweeps of dual BCD: sweeps [1, 35, 114, 975, 2332] (gap reached in 5 of 5), nonzero bands (0, 15) of 15, largest factor spread 0.726, largest gap/(1+‖r‖²) 1e-12, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0166, kept 0 against √(2 gap) ≤ 6.48e-05
PASS CD p = 32, general weights, λ = 2.5, 5 instances (seeded): sweeps of dual BCD: sweeps [1, 1, 10, 2687, 3955] (gap reached in 5 of 5), nonzero bands (0, 26) of 31, largest factor spread 0.781, largest gap/(1+‖r‖²) 1e-12, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0242, kept 0 against √(2 gap) ≤ 0.000142
PASS CD p = 64, general weights, λ = 2.5, 5 instances (seeded): sweeps of dual BCD: sweeps [13, 20, 328, 2760, 100000] (gap reached in 4 of 5), nonzero bands (5, 63) of 63, largest factor spread 0.925, largest gap/(1+‖r‖²) 2.2e-10, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0295, kept 0 against √(2 gap) ≤ 0.00387
PASS CD p = 128, general weights, λ = 2.5, 5 instances (seeded): sweeps of dual BCD: sweeps [9, 17, 620, 1358, 7591] (gap reached in 5 of 5), nonzero bands (97, 126) of 127, largest factor spread 0.965, largest gap/(1+‖r‖²) 1e-12, one pass off by ‖x₁ − x‖/‖r‖ up to 0.0308, kept 0 against √(2 gap) ≤ 0.000529
all statements hold
```

### `check_inventory.py`

```text
76 citations, 0 failure(s)
```
