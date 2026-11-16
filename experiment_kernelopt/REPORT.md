# ProximalBase numerical kernels: optimization study

Status: complete. Nothing was merged, and nothing was changed in ProximalSolvers.

**This branch is a reading copy** (published 2026-11-16): the report, the scripts and the summary tables. The raw
results (`results/`, 205 MB of CSV, provenance and Slurm outputs) are not in it; they are kept in the study's
local branch and on the cluster. The four patches are draft pull requests, each the measured commit plus a
merge of master `b8ddeaf3`:

| Patch | Draft PR | Recommendation |
|:--|:--|:--|
| D3: skip the second scan of a zero group | #966 | implement |
| V: one loop for NormL1 and NormL2 values of 16 to 1024 entries | #967 | implement if the last-bit change is acceptable |
| B925: #925's design for NormL1 step arrays | #968 | one revision, then propose |
| A3: select-based soft threshold on the ordinary NormL1 path | #969 | do not merge under the current policy |

Baselines: the study started from ProximalBase master `50db4b1ef376cbe01831c3dd019b6be0027db0b6` (after #918), and
§§4–8.5 are measured against it. Master then moved to `18b27149` (#909 ElasticNet, #922, #923, #925, which
resolved #910, and #931); the proposed patches were rebased on it and §8.6 repeats their comparison there.
Primary target Julia 1.13.4 (the latest stable release, 2026-11-13), secondary Julia 1.10.13 (LTS). All timings
in this report are from USC Discovery unless a line says "workstation".

## 1. Summary

**Where ProximalBase stands.** Least squares, NormL2 and group shrinkage of about a thousand entries, and the
structure of the ℓ1 loops are at the floor of check-free kernels written for this study (§4). ProximalOperators,
BLAS `nrm2` and Blue's algorithm are not faster where they are as accurate (§4, §6). What is left is small and
specific: the four patches below change 15 to 34 lines of source each.

**Recommendations (§9).**

| | Patch | Recommendation | Gain | Needs from the maintainer |
|--:|:--|:--|:--|:--|
| 1 | D3: skip the second scan of a zero group | Implement | 0.83–0.95 on zero and sparse groups | acceptance of 1.10–1.15× for zero groups of 1 to 5 entries at a threshold near `floatmin` (0.3 ns) |
| 2 | V: one loop for NormL1 and NormL2 values of 16 to 1024 entries | Implement if the last-bit change is acceptable | 0.62–0.87 | a numerical ruling: at most 1 eps different inside the window |
| 3 | B925: #925's design for NormL1 step arrays | One revision, then propose | 0.10–0.45 | #925's costs at 1 to 9 entries; the reviewer's short-array bypass is not written |
| 4 | A3: select-based soft threshold on the ordinary NormL1 path | **Do not merge under the current policy**; ready if ruled otherwise | 0.65–0.97 for dense NormL1 and ElasticNet `prox!`; 3× on Julia 1.10 scalar paths | reproducible slowdowns of four kinds, below |

**A3 is the largest gain of the study and I do not recommend it under the 5% rule.** It was the lead proposal
until the last campaign. The confirmation on the new master then showed it 1.20–1.29× slower in every process
for ElasticNet at 2^20 Float32 entries on Zen 3 with Julia 1.10, and unstable from process to process with
Julia 1.13.4. Controlled probes then showed a placement effect (§8.7): when the output starts 16 to about 256
bytes after the input modulo 4096 and the memory is on huge pages, most fast loops slow down on every CPU, and
on Zen 3 A3's loop loses more than master's slower one, up to 1.3× for NormL1 and 1.6× for ElasticNet at 16K
entries and more. Julia's allocator often puts two consecutively allocated arrays of a power-of-two size
there, and the cluster runs with huge pages always on. Elsewhere, and on 4 KiB pages, A3 is 0.65–0.87 on
Zen 3. On the Intel nodes master pays the same penalty and A3 stays equal or faster at nearly every offset.
The mechanism is not established, and the formal runs did not record their arrays' addresses, so this is the
likely explanation of those runs, not a proven one. A3's other costs are 1 to 2.5 ns at lengths up to 24 on
Julia 1.10 and a few cells of 1 to 100 entries on Julia 1.13 (§9).

**Two findings about benchmarks that matter beyond these patches (§8.7, §8.8, §10).**

- Benchmarks that allocate the input and the output one after the other with power-of-two sizes often sit in
  that placement band. In it, on huge pages, master's own vectorized loops (NormL1, ridge and NormL2 `prox!`)
  are up to 1.3 to 2 times slower on Zen 3 than at offset 0, and up to 5.8 times on the Intel nodes.
- On Julia 1.10 master's scalar soft threshold has two data-dependent branches. A benchmark that repeats one
  input lets the Zen 3 predictor learn them: 10.5 µs for 6,553 stored entries on a repeated input, 41–50 µs
  when the values change. A3 has no such branches (about 14 µs), so it reads 1.15–1.22× slower in PB's own
  `ridge910.jl` on that machine and is 3 to 3.6 times faster on values that change at random.

**Julia versions (§5).** PB costs the same on Julia 1.13.4 and 1.10.13 for most operations (median ratio
1.02–1.06). The exceptions are the weighted NormL1 value, which is not vectorized on 1.10 (1.8–14× slower;
draft PR #155 fixes it, but is 1.14–1.28× slower than master on the AVX-512 node with Julia 1.13.4), and the
branches above.

**Solvers (§8.3).** No patch changes a solver iteration of a regression workload by more than 1%: the gradient
dominates. The gains matter where the proximal step is the iteration.

**Reviews.** An independent Codex reviewer checked the benchmark methodology once, the patches three times, and
the placement and repeated-input findings once. The patch reviews confirmed the numerics of A3, V and D3. The
last (`notes/REVIEW-placement-round4.md`) supports the recommendation on A3, found my first claim that page
size does not matter to be wrong, rated the allocator statement as a frequent outcome and not a rule, and
recomputed the tables; its corrections are applied in §8.7 and §8.8.

## 2. What was measured, and how

### 2.1 Levels

1. **Inner kernel.** Minimal standalone loops (`kernels/kernels.jl`): no validation, no recovery, the threshold
   precomputed. They are the floor that an ordinary path can approach, not candidates: most of them fail
   exceptional inputs (§6).
2. **Library API.** The public operations of ProximalBase (PB), of ProximalOperators.jl 0.16.1 (PO), of
   LinearAlgebra and of BLAS, with their own validation and return values. PO's `prox!` returns the function
   value at the proximal point and PB's returns the output array; PO is therefore compared with PB's
   `value_and_prox!` as well as with `prox!`.
3. **Solver.** Fixed-iteration proximal gradient, accelerated proximal gradient, coordinate descent and block
   coordinate descent loops written against PB's public interface (`bench/solvers.jl`), one implementation per
   source tree. ProximalSolvers and its PB pin were not touched.

### 2.2 Harness

`bench/run.jl` loads every source tree into its own module in one process (as PB's own harnesses do), builds the
cases, checks every implementation's output against the reference implementation and against a BigFloat
reference that uses no package code, and times them.

- Inputs and outputs are shared by all implementations of a case, so array placement cannot differ between
  them. A case therefore measures one placement of its arrays per process, and placement matters more than this
  design assumed: §8.7.
- Timed closures capture function values and concrete arguments, so every call is statically dispatched. The
  call sits behind a `@noinline` barrier so that a pure kernel on unchanged inputs is not hoisted.
- A timing is the minimum of 3 batches of about 2 ms; a case has 21 rounds. Rounds come in pairs: a rotation of
  the implementation order and its reversal.
- The reference tree is compiled a second time in every process (`baseC`), the A/A control.
- A formal comparison is 6 fresh processes per stratum (CPU kind × Julia version), each with its own tree
  compilation order and with the control first or last. **The process is the unit of replication.** A case is
  called *faster* when its per-process median ratio is below 0.95 in every process, *slower* when it is above
  1.05 in every process, and *unresolved* when the A/A control leaves [0.98, 1.02] in some process.
- One pinned core of an exclusive node, one Julia thread, one BLAS thread, `--check-bounds=no`,
  `--startup-file=no`.

### 2.3 Independent review of the methodology

A Codex reviewer (gpt-6.1-sol, high effort) reviewed the harness before the formal runs
(`notes/REVIEW-methodology.md`). Three findings were blockers for a ±5% decision and were fixed before the
confirmatory campaigns (`formal2`, `screen2` and everything after them):

| Finding | Fix |
|:--|:--|
| A NaN in an output passed the tolerance check (`NaN > rtol` is false), and a failed check did not fail the job | Class mismatches, shape mismatches and tuple arity give `Inf`; the check is `!(d <= rtol)`; a difference exits non-zero and the run gets no `DONE`. All trees and external implementations were re-checked with the strict comparison on both Julia versions: no differences and no active-set mismatches. |
| With an even number of implementations, rotation followed by reversal left the order unbalanced (two implementations were always timed in the same order) | Rounds are now pairs (rotation, its reversal); process orders are pairs as well. |
| Pooled rounds are not independent observations, and an A/A column is not a decision rule | Per-process medians are reported; the decision rule above uses every process and the control band. |

Findings that were not fixed, and what they mean for the conclusions:

- Kernel-level ratios (level 1) are not production speedups: the kernels skip validation and recovery. They are
  used only to say how far an ordinary path is from its floor.
- The output tolerances scale with the length (`4 n eps`) and are loose for long Float32 vectors. Bit-level claims
  in this report come from the separate differential checks of §7, not from the benchmark's tolerance.
- Frequency governor, transparent huge pages and SMT state are recorded in `provenance.txt` but not controlled.
  Neither is the placement of the arrays, which the probes of §8.7 control instead.
- Cases cover dense arrays, equal-size groups with the default weights, and contiguous and permuted index groups.
  Sparse, block and symmetric iterates are covered only by PB's own harnesses (§8.4).

### 2.4 Campaigns

| Tag | Purpose | Harness | CPU kinds | Jobs |
|:--|:--|:--|:--|:--|
| `screen1` | PB against kernels and external libraries, both Julia versions | v1 | Xeon E5-2640 v4 (exclusive), Xeon Silver 4116 (exclusive and shared), EPYC 7513 (shared) | 14273951, 14273952, 14273955, 14273956 |
| `formal1` | first tree comparison (A, B, C, C2, D, P155, F), 3 processes | v1 | Xeon E5-2640 v4, Xeon Silver 4116 | 14275001, 14275003, 14275004 |
| `formal2` | tree comparison with A2 and the combined tree, 6 processes | v2 | Xeon E5-2640 v4, Xeon Silver 4116 | 14275541, 14275542, 14275547 |
| `screen2` | PB against kernels and external libraries | v2 | Xeon E5-2640 v4, EPYC 7513 | 14275543, 14275546 |
| `formal4` | the revised trees after the patch reviews (A3, V, D3, B2 and FINAL2 = A3 + V + D3), 6 processes | v2 | EPYC 7513, Xeon E5-2640 v4, Xeon Silver 4116 | 14277322, 14277326, 14277323, 14277327, 14277330 |
| `tiny1` | the same trees on 1 to 100 entries and on zero groups below D3's guard, 6 processes, both Julia versions | v2 | EPYC 7513, Xeon E5-2640 v4 | 14277951, 14277952 |
| `native3` | PB's own paired harnesses on FINAL2, both role orders | PB `benchmark/` | Xeon E5-2640 v4 | 14277324, 14277325, 14277328, 14277329 |
| `m2` | new master `18b27149`: trees A3, V, D3, B925 and FINAL rebased, 6 processes; PB's harnesses. A3 and FINAL of this campaign carry the ElasticNet edit that was later withdrawn (§7.1), so only its V, D3 and B925 columns and its harness runs that do not call ElasticNet are used | v2, PB `benchmark/` | Xeon E5-2640 v4, Xeon Silver 4116 | 14281655, 14281658, 14281662 (`formal5`); 14281656, 14281657, 14281659, 14281660, 14281668, 14281670 (`native4`); 14281661 (`screen3`) |
| `m3` | new master, final source of A3 and FINAL: tree comparison, tiny inputs, ElasticNet against bare kernels, `benchmark/ridge910.jl` and the NormL1 harnesses | v2, PB `benchmark/` | EPYC 7513, Xeon E5-2640 v4, Xeon Silver 4116 | 14282587, 14282590, 14282594, 14282596, 14282598 (`formal6`); 14282588, 14282591, 14284284, 14284285 (`tiny3`); 14282593 (`screen4`); 14282595, 14282597, 14284530–14284533 (`native5`; 14282589 and 14282592 were cancelled, and the `ridge910` part of 14282597 failed on a stale launcher and was rerun as 14284532) |
| `big` | the million-entry cases of `formal6` again, 6 processes | v2 | EPYC 7513 (nodes a03-04 and b04-18) | 14284526, 14284527, 14284528 |
| `probe1`–`probe7` | array placement (§8.7) and repeated inputs (§8.8): master and a candidate on the same arrays at controlled page offsets and page sizes, 2 to 3 processes per node and Julia version | `bench/layout_probe.jl`, `bench/offset_probe.jl`, `bench/sparse_probe.jl` | EPYC 7513 (a03-04, b09-17, a04-18), Xeon E5-2640 v4, Xeon Silver 4116 | 14286886–14286890 (`probe1`, layouts); 14286935, 14286936 (`probe2`, offset sweep, Intel nodes); 14286974, 14286975 (`probe3`, sparse inputs, one set of arrays per tree); 14287057–14287060 (`probe5`, offset sweep with master's other loops; 14287056 cancelled); 14287080, 14287082 (`probe6`, sparse inputs, shared arrays; 14287081 cancelled); 14287177–14287179 (`probe7`, offset sweep with B925). Not used: 14286934 and 14286990–14286992 ran a faulty version of the sweep (its closures shared a captured variable, so every library row timed one function), and 14286934 overwrote the Zen 3 output of 14286933; their files are in `results/m3/probe-invalid/` with a note. |

Two campaigns were submitted and cancelled when a review changed the trees before they finished (`formal3`,
`native2`); they are not used.

CPU kinds: AMD EPYC 7513 (Zen 3, AVX2), Intel Xeon E5-2640 v4 (Broadwell, AVX2), Intel Xeon Silver 4116
(Skylake-SP, AVX-512). Raw results, Slurm scripts, job outputs and provenance files are under `results/<tag>/`.

## 3. Kernel inventory

Sources read (versions, commits and licenses in `notes/external_inventory.md`): ProximalOperators.jl, Julia's
LinearAlgebra and Base for 1.13.4 and 1.10.13, OpenBLAS 0.3.30 and 0.3.23 (the versions the two Julia releases
bundle), reference LAPACK (`dnrm2.f90`, `dlassq.f90` and the older `dnrm2.f`), glmnet 5.2, grpreg, Lasso.jl,
skglm and celer. glmnet (GPL-2) and grpreg (GPL-3) can contribute ideas only. No code was copied from any of
them; the kernels named after a library are written from the description of its algorithm.

### A. Soft thresholding and the weighted ℓ1 norm

| | Per entry | Signed zero below the threshold | NaN input |
|:--|:--|:--|:--|
| PB `shrink` | `max(v - c, 0) + min(v + c, 0)` | +0 | NaN |
| PO `NormL1` | two compares, two selects, an addition, with the value summed in the same loop | +0 | NaN |
| glmnet, grpreg, celer | branch on `\|z\| - λ`, then copysign or ±λ | +0 | **0** |
| skglm `ST_vec` | `sign(x) max(\|x\| - λ, 0)` | **-0** | NaN |
| Lasso.jl | branch and select | +0 | NaN |

PB forms the threshold `γλ0·w` once per entry with a range test on `γλ0` that is loop-invariant for a scalar
step; PO precomputes `λ.*w` (and allocates when given per-entry weights). PB's semantics are the strictest of
the set. Its *formula* is not the cheapest: §7.1.

### B. Euclidean norm and group shrinkage

| | Ordinary path | Overflow and underflow of the squares |
|:--|:--|:--|
| PB (`_indexed_sumabs2`, `_l2_shrink!`) | one SIMD sum of squares, a range test of the sum, one sqrt, one division, one scaling pass | recomputed in power-of-two units on a cold path |
| Julia `norm`, n ≥ 32 | OpenBLAS `dnrm2`: the x87 kernel `nrm2.S` on every x86-64 target, 80-bit accumulation, scalar | safe by the extended range |
| Julia `norm`, n < 32 | `generic_norm2`: a maximum pass, then a sequential sum | scaled when the maximum is extreme |
| LAPACK ≥ 3.10 `dnrm2` | one branchy pass with three accumulators (Blue, Anderson) | safe |
| PO `NormL2`, `NormL21` | `norm` (as above), then a broadcast scale | as `norm`; 0/0 at a zero level or step |
| glmnet (Eigen `norm`), grpreg | plain sum of squares | none |

PB, PO and glmnet all use one square root and one division per group; grpreg divides per entry.

### C. Least squares

PB's value is a one-loop SIMD sum of squares up to 2^17 entries and Julia's pairwise sum above, with a range
test on the bits of the sum. Its gradient is `d = z - Y`, one `gemv` with `Xᵀ`, and the value from `d`. Its
proximal operator caches the thin SVD of `X` on first use and then costs two products with `V` for any step.
PO's `LeastSquares` caches a Cholesky factor of `XᵀX + I/γ` (or of `XXᵀ + I/γ` with the Woodbury identity for
wide `X`), refactorizes whenever `γ` changes, and evaluates the loss at the output on every `prox!`.

### D. Ridge and elastic net

PB's ridge proximal operator multiplies by a weight formed once per call; PO, glmnet and Lasso.jl divide per
entry or per coordinate update. ElasticNet reached master during the study (#909), and #925 resolved #910 with
a loop specialized to the ordinary branches for step arrays. No ridge or ElasticNet source is changed by any
patch proposed here: the one ElasticNet edit that was tried (A3's formula at ElasticNet's two direct `shrink`
calls) was withdrawn after review (§7.1). ElasticNet is measured on the new master in §8.6.

## 4. Measured comparison

One exclusive node per column; one process, 21 rounds, harness v2 for the Zen 3 and Broadwell columns
(`screen2`), v1 for Skylake-SP (`screen1`, which does not have the A3 kernel). A row is the geometric mean, over
the sizes named, of the per-round ratio of the implementation's time to ProximalBase's public operation; below 1
is faster than PB. PB's own rows give its time range over those sizes. The full tables, with Float32 and the
Julia 1.10 columns of every machine, are `notes/tables/screen-{l1,l2,group,ls}.md`; none of the implementations
allocates except ProximalOperators with per-entry weights.

### 4.1 Soft thresholding and ℓ1 values (Float64)

| operation | implementation | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Skylake-SP, 1.13 |
|:--|:--|--:|--:|--:|--:|
| Float64 prox, scalar step, n = 128…16384 | **ProximalBase** (time) | 26.6–3.38e+03 ns | 31.4–3.83e+03 ns | 58.3–7.32e+03 ns | 36.3–4.88e+03 ns |
|  | bare loop, PB's formula | 1.00 | 1.00 | 0.99 | 0.98 |
|  | bare loop, select max/min (A) | 0.83 | 0.72 | 0.82 | 0.85 |
|  | bare loop, A3's formula | 0.67 | 0.59 | 0.66 |  |
|  | bare loop, plain clamp | 0.56 | 0.50 | 0.56 | 0.62 |
|  | ProximalOperators | 1.20 | 1.56 | 1.02 | 1.36 |
| Float64 prox, weights | **ProximalBase** (time) | 28.2–3.42e+03 ns | 37.5–4.64e+03 ns | 60.5–8.25e+03 ns | 45.5–5.98e+03 ns |
|  | bare loop, PB's formula | 0.99 | 0.98 | 1.00 | 1.00 |
|  | bare loop, A3's formula | 0.81 | 0.62 | 0.77 |  |
|  | ProximalOperators | 4.07 | 5.21 | 6.80 | 6.12 |
| Float64 prox, step per entry | **ProximalBase** (time) | 149–1.92e+04 ns | 226–2.76e+04 ns | 311–3.88e+04 ns | 250–3.05e+04 ns |
|  | bare loop, PB's formula | 0.18 | 0.16 | 0.20 | 0.18 |
|  | bare loop, A3's formula | 0.15 | 0.10 | 0.15 |  |
| Float64 value | **ProximalBase** (time) | 19–1.3e+03 ns | 22.2–1.35e+03 ns | 26.3–2.46e+03 ns | 30–2.13e+03 ns |
|  | bare one-loop sum | 0.67 | 0.62 | 0.60 | 0.70 |
|  | `sum(abs, x)` | 0.92 | 0.94 | 0.94 | 0.95 |
|  | BLAS `asum` | 1.23 | 1.13 | 0.72 | 0.85 |
|  | ProximalOperators | 1.27 | 1.14 | 0.75 | 0.87 |
| Float64 value, weights | **ProximalBase** (time) | 27.6–2.87e+03 ns | 133–1.79e+04 ns | 49.3–5.79e+03 ns | 31.2–3.82e+03 ns |
|  | bare one-loop sum | 0.52 | 0.10 | 0.44 | 0.79 |
|  | ProximalOperators | 5.38 | 1.08 | 3.81 | 5.73 |
| Float64 value and prox | **ProximalBase** (time) | 38.2–4.54e+03 ns | 47.8–4.95e+03 ns | 77.3–9.53e+03 ns | 59.2–6.94e+03 ns |
|  | bare, two passes | 0.95 | 0.94 | 0.95 | 0.92 |
|  | bare, one fused pass | 0.90 | 0.91 | 0.91 | 0.84 |
|  | bare fused pass, plain clamp | 0.58 | 0.49 | 0.54 | 0.57 |
|  | ProximalOperators | 0.89 | 1.13 | 0.80 | 0.92 |
| Float64 prox, n = 2^20 (out of cache) | **ProximalBase** (time) | 2.17e+05 ns | 2.47e+05 ns | 5.69e+05 ns | 8.83e+05 ns |
|  | bare loop, PB's formula | 1.00 | 1.00 | 1.00 | 1.00 |
|  | bare loop, A3's formula | 0.84 | 0.75 | 0.98 |  |
|  | ProximalOperators | 1.13 | 1.57 | 1.02 | 1.02 |

- **Scalar step: PB's public `prox!` is its bare loop** (1.00): validation and the threshold product cost
  nothing measurable. The formula is where the time goes: the plain clamp is 0.50–0.62×, A3's formula 0.59–0.70×.
- **A step per entry is PB's largest gap**: the bare loop is 5–10× faster, because PB's loop is not vectorized.
- **Values:** a one-loop sum is 0.52–0.70× of `sum(abs, x)`; the weighted value is 2× the bare sum on Julia 1.13
  and 8–10× on Julia 1.10 (§5).
- **ProximalOperators** is slower than PB everywhere except the unweighted value on Intel (it calls BLAS `asum`,
  which is slower than PB on Zen 3), and 4–7× slower with per-entry weights, where it allocates.
- **Out of cache nothing differs**: at 2^20 entries every variant is at 1.00.
- Float32 has the same picture (`notes/tables/screen-l1.md`): plain clamp 0.46–0.62×, A3's formula 0.60–0.72×,
  step arrays 0.06–0.12×.

### 4.2 Euclidean norm and its shrinkage (Float64)

| operation | implementation | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Skylake-SP, 1.13 |
|:--|:--|--:|--:|--:|--:|
| Float64 value, n = 2…10 | **ProximalBase** (time) | 6.02–8.79 ns | 5.75–9 ns | 7.67–11.7 ns | 7.78–12.9 ns |
|  | bare one-loop sum of squares | 0.48 | 0.50 | 0.50 | 0.52 |
|  | `LinearAlgebra.norm` | 1.45 | 1.66 | 1.90 | 1.84 |
|  | BLAS `nrm2` | 1.37 | 1.62 | 0.99 | 0.98 |
|  | three-accumulator pass (LAPACK 3.10 style) | 1.16 | 1.25 | 1.13 | 1.29 |
|  | ProximalOperators | 1.45 | 1.67 | 1.88 | 1.77 |
| Float64 value, n = 32…1024 | **ProximalBase** (time) | 17.5–85.8 ns | 21.7–92.3 ns | 21.8–151 ns | 23.7–114 ns |
|  | bare one-loop sum of squares | 0.48 | 0.43 | 0.38 | 0.51 |
|  | `sqrt(sum(abs2, x))` | 0.85 | 0.91 | 0.90 | 0.90 |
|  | `LinearAlgebra.norm` | 2.95 | 2.57 | 1.29 | 1.69 |
|  | BLAS `nrm2` | 2.91 | 2.54 | 1.27 | 1.66 |
|  | `sqrt(dot(x, x))` | 0.75 | 0.67 | 0.62 | 0.75 |
|  | three-accumulator pass (LAPACK 3.10 style) | 4.09 | 3.49 | 2.82 | 5.01 |
|  | ProximalOperators | 2.95 | 2.57 | 1.29 | 1.69 |
| Float64 value, n = 65536 | **ProximalBase** (time) | 5.63e+03 ns | 7e+03 ns | 1.31e+04 ns | 7.23e+03 ns |
|  | bare one-loop sum of squares | 0.83 | 0.77 | 0.97 | 0.79 |
|  | `LinearAlgebra.norm` | 6.38 | 5.14 | 1.63 | 3.40 |
|  | BLAS `nrm2` | 6.39 | 5.14 | 1.63 | 3.40 |
|  | three-accumulator pass (LAPACK 3.10 style) | 9.58 | 7.71 | 4.43 | 12.13 |
|  | ProximalOperators | 6.38 | 5.14 | 1.63 | 3.40 |
| Float64 prox, n = 32…1024 | **ProximalBase** (time) | 13–152 ns | 14.5–155 ns | 17.8–202 ns | 19.7–223 ns |
|  | bare kernel (no checks) | 0.93 | 0.99 | 0.93 | 0.95 |
|  | `norm`, then broadcast | 3.23 | 3.14 | 1.85 | 1.86 |
|  | `nrm2`, then broadcast | 3.18 | 3.09 | 1.83 | 1.82 |
|  | ProximalOperators | 3.17 | 3.09 | 1.82 | 1.81 |
| Float64 prox of the zero vector, n = 32…1024 | **ProximalBase** (time) | 10.4–169 ns | 11.2–169 ns | 17.4–261 ns | 19.2–220 ns |
|  | bare kernel (no checks) | 0.86 | 0.92 | 0.67 | 0.69 |
|  | ProximalOperators | 3.69 | 3.74 | 1.67 | 1.90 |

- **PB's shrinkage is at its floor** from 32 entries (0.93–0.99); `norm`-based shrinkage, which is what
  ProximalOperators does, is 1.8–3.2× slower, because Julia sends vectors of 32 or more entries to OpenBLAS's x87
  `nrm2`.
- **The value has a gap**: the bare sum of squares is 0.38–0.51× of PB's value for 32–1024 entries, and Base's
  own `sum(abs2, x)` is only 0.85–0.92×. Most of PB's remaining cost at these sizes is the range checks and the
  call into Base's reduction; candidate V addresses the second part.
- The three-accumulator pass of LAPACK 3.10, written in Julia, is 2.8–5× slower than PB's value: a safe one-pass
  norm is not competitive with "plain sum, then check, then a cold recomputation".
- A zero vector costs PB a second pass (0.74–0.80× for the bare kernel in the prox): candidate D3.

### 4.3 Group shrinkage (Float64, 4096 entries in equal groups, default weights)

| operation | implementation | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Skylake-SP, 1.13 |
|:--|:--|--:|--:|--:|--:|
| Float64 prox, groups of 2, range, allshrink | **ProximalBase** (time) | 1.34e+04 ns | 1.78e+04 ns | 2.23e+04 ns | 2.51e+04 ns |
|  | bare kernel (no checks) | 0.91 | 0.60 | 0.86 | 0.84 |
|  | ProximalOperators | 0.76 | 0.56 | 0.60 | 0.60 |
| Float64 prox, groups of 32, range, allshrink | **ProximalBase** (time) | 1.57e+03 ns | 1.82e+03 ns | 2.4e+03 ns | 2.65e+03 ns |
|  | bare kernel (no checks) | 0.84 | 0.76 | 0.83 | 0.81 |
|  | ProximalOperators | 1.73 | 1.61 | 1.49 | 1.86 |
| Float64 prox, groups of 32, range, sparse | **ProximalBase** (time) | 1.64e+03 ns | 1.79e+03 ns | 2.3e+03 ns | 2.62e+03 ns |
|  | bare kernel (no checks) | 0.71 | 0.69 | 0.76 | 0.69 |
|  | ProximalOperators | 1.66 | 1.67 | 1.49 | 1.85 |
| Float64 prox, groups of 1024, range, allshrink | **ProximalBase** (time) | 686 ns | 690 ns | 1.45e+03 ns | 1.26e+03 ns |
|  | bare kernel (no checks) | 1.01 | 0.96 | 1.01 | 1.01 |
|  | ProximalOperators | 5.41 | 5.37 | 2.88 | 5.04 |
| Float64 prox, groups of 1024, range, sparse | **ProximalBase** (time) | 740 ns | 755 ns | 1.59e+03 ns | 1.27e+03 ns |
|  | bare kernel (no checks) | 0.90 | 0.87 | 0.89 | 0.87 |
|  | ProximalOperators | 5.01 | 4.90 | 2.62 | 4.99 |
| Float64 value, groups of 2, range, allshrink | **ProximalBase** (time) | 8.3e+03 ns | 9.45e+03 ns | 1.15e+04 ns | 1.24e+04 ns |
|  | bare kernel (no checks) | 0.62 | 0.57 | 0.57 | 0.67 |
|  | ProximalOperators | 0.59 | 0.53 | 0.39 | 0.45 |
| Float64 value, groups of 32, range, allshrink | **ProximalBase** (time) | 904 ns | 1.1e+03 ns | 1.27e+03 ns | 1.41e+03 ns |
|  | bare kernel (no checks) | 0.70 | 0.63 | 0.72 | 0.70 |
|  | ProximalOperators | 2.12 | 1.68 | 1.68 | 2.07 |
| Float64 value, groups of 32, range, sparse | **ProximalBase** (time) | 1.11e+03 ns | 1.49e+03 ns | 2.3e+03 ns | 2.53e+03 ns |
|  | bare kernel (no checks) | 0.56 | 0.46 | 0.40 | 0.38 |
|  | ProximalOperators | 1.70 | 1.25 | 0.89 | 1.13 |
| Float64 value, groups of 1024, range, allshrink | **ProximalBase** (time) | 298 ns | 299 ns | 407 ns | 412 ns |
|  | bare kernel (no checks) | 0.98 | 0.97 | 0.98 | 0.98 |
|  | ProximalOperators | 11.05 | 10.98 | 8.69 | 12.99 |
| Float64 value, groups of 1024, range, sparse | **ProximalBase** (time) | 418 ns | 415 ns | 630 ns | 597 ns |
|  | bare kernel (no checks) | 0.70 | 0.70 | 0.62 | 0.66 |
|  | ProximalOperators | 7.85 | 7.91 | 5.61 | 8.96 |
| Float64 block_sweep, groups of 2, range, allshrink | **ProximalBase** (time) | 2.27e+04 ns | 2.17e+04 ns | 3.78e+04 ns | 4.14e+04 ns |
|  | bare kernel (no checks) | 0.47 | 0.47 | 0.55 | 0.50 |
| Float64 block_sweep, groups of 32, range, allshrink | **ProximalBase** (time) | 2.23e+03 ns | 2.37e+03 ns | 3.39e+03 ns | 3.46e+03 ns |
|  | bare kernel (no checks) | 0.57 | 0.68 | 0.61 | 0.61 |
| Float64 block_sweep, groups of 1024, range, allshrink | **ProximalBase** (time) | 692 ns | 704 ns | 1.42e+03 ns | 1.13e+03 ns |
|  | bare kernel (no checks) | 0.99 | 0.99 | 1.00 | 1.01 |

- **Groups of 1024: PB is at the floor** (0.96–1.01) unless groups are zero (0.87–0.90 for the prox, 0.61–0.70
  for the value: the second pass).
- **Small groups: 15–45% above a check-free kernel** in the prox and up to 2× in the value. §7.6 shows that this
  is the per-group range checks (25–45 instructions), not the loop.
- **The compact `prox_block!` is 1.5–2.2× a bare block kernel for groups up to 32**: the cost of a call that is
  not inlined plus its argument checks (candidate F, §7.5).
- **ProximalOperators' `NormL21`** (groups as matrix columns, no weights) is 0.53–0.77× of PB for groups of 2,
  because it does no checks, and 1.4–13× slower from 32 entries, because its sums are sequential.
- Index-vector (permuted) groups are bound by the gathers: PB is within 3% of the bare kernel from 32 entries
  (`notes/tables/screen-group.md`).

### 4.4 Least squares (Float64)

| operation | implementation | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Skylake-SP, 1.13 |
|:--|:--|--:|--:|--:|--:|
| Float64 value at a residual, n = 100…100000 | **ProximalBase** (time) | 10–8.38e+03 ns | 11.1–8.43e+03 ns | 13.5–1.94e+04 ns | 15.2–8.73e+03 ns |
|  | bare one-loop sum | 0.96 | 0.96 | 0.97 | 0.98 |
|  | `sum(abs2, r)/2` | 1.17 | 1.18 | 1.58 | 1.23 |
|  | `dot(r, r)/2` | 1.21 | 1.16 | 1.33 | 1.52 |
|  | `nrm2(r)^2/2` | 7.10 | 6.81 | 2.84 | 3.95 |
| Float64 value at a predictor, n = 100…100000 | **ProximalBase** (time) | 12.1–1.73e+04 ns | 13.5–1.75e+04 ns | 30.1–3.62e+04 ns | 20.9–7.03e+04 ns |
|  | bare one-loop sum | 1.00 | 1.01 | 0.91 | 0.97 |
| Float64 value at β (one product with X) | **ProximalBase** (time) | 187–1.01e+05 ns | 177–1.02e+05 ns | 268–2.06e+05 ns | 468–4.37e+05 ns |
|  | bare kernel (no checks) | 0.99 | 0.99 | 0.99 | 0.98 |
|  | ProximalOperators | 1.10 | 1.11 | 1.04 | 1.07 |
| Float64 value and gradient at β | **ProximalBase** (time) | 363–2e+05 ns | 357–2e+05 ns | 559–3.94e+05 ns | 945–8.77e+05 ns |
|  | bare: subtract, `gemv`, sum | 1.00 | 1.00 | 1.00 | 0.99 |
|  | bare: fused subtract and sum | 1.00 | 0.99 | 0.99 | 0.99 |
|  | bare: residual by `gemv` | 1.00 | 1.00 | 0.99 | 1.00 |
|  | ProximalOperators | 1.01 | 1.00 | 1.01 | 1.01 |
| Float64 value and gradient at a predictor | **ProximalBase** (time) | 203–9.88e+04 ns | 197–9.94e+04 ns | 308–1.9e+05 ns | 315–3.69e+05 ns |
|  | bare: subtract, `gemv`, sum | 1.00 | 0.98 | 0.99 | 0.99 |
|  | bare: fused subtract and sum | 0.99 | 0.98 | 0.98 | 0.99 |
| Float64 prox, fixed step, tall X | **ProximalBase** (time) | 545–5.03e+04 ns | 545–5.11e+04 ns | 896–9.45e+04 ns | 1.43e+03–2.2e+05 ns |
|  | cached Cholesky | 2.86 | 2.83 | 2.42 | 1.97 |
|  | ProximalOperators | 3.91 | 3.95 | 3.18 | 3.80 |
| Float64 prox, fixed step, wide X (200×1000) | **ProximalBase** (time) | 4.16e+04 ns | 4.19e+04 ns | 7.91e+04 ns | 1.71e+05 ns |
|  | cached Cholesky | 1.56 | 1.56 | 1.46 | 1.35 |
|  | ProximalOperators | 1.85 | 1.97 | 1.65 | 1.64 |
| Float64 prox, the step changes at every call | **ProximalBase** (time) | 1.09e+03–9.77e+04 ns | 1.1e+03–9.71e+04 ns | 1.71e+03–1.89e+05 ns | 2.87e+03–4.41e+05 ns |
|  | cached Cholesky | 14.93 | 14.70 | 11.36 | 15.26 |
|  | ProximalOperators | 26.01 | 20.26 | 17.85 | 20.28 |
| Float64 construction and first prox | **ProximalBase** (time) | 5.6e+05–1.32e+08 ns | 5.51e+05–1.28e+08 ns | 8.66e+05–1.85e+08 ns | 1.29e+06–2.67e+08 ns |
|  | cached Cholesky | 0.07 | 0.07 | 0.07 | 0.08 |
|  | ProximalOperators | 0.08 | 0.08 | 0.08 | 0.09 |

- **Values and gradients are at the floor on every machine** (0.94–1.00), including against fused variants; the
  two `gemv` calls are the cost, and ProximalOperators' gradient is the same two calls (1.00–1.02).
- `sum(abs2, r)`, `dot` and `nrm2` are all slower than PB's value (1.1–6.5×).
- **The proximal operator: PB's cached SVD is the fastest per call** (a cached Cholesky is 1.8–2.8× slower for
  tall designs and 1.35–1.46× for the wide one; ProximalOperators 3.2–4.2×), and 11–24× faster when the step
  changes between calls, since nothing is refactorized.
- **Its one-time cost is the weakness**: construction plus the first prox is 12–14× the Cholesky route. With the
  per-call times above, the SVD pays for itself after about 75 calls when the step varies and about 500 calls at
  a fixed step (200 × 50). §6 gives the accuracy side.

## 5. Julia 1.13.4 against 1.10.13

Same node, same process structure, the two versions one after the other (`notes/tables/versions-*.md`).

| | Zen 3 | Broadwell | Skylake-SP |
|:--|--:|--:|--:|
| PB time on 1.10 / on 1.13, median over 528 cases | 1.06 | 1.02 | 1.03 |
| geometric mean | 1.19 | 1.13 | 1.16 |
| cases above 1.30 | 87 | 51 | 103 |
| cases below 0.77 | 2 | 6 | 3 |

Most operations cost the same on the two versions. The exceptions are concentrated in four places, and in each
the bare kernel does not have the problem (its 1.10/1.13 ratio is in the last column), so the cause is how
PB's loop is written, not the compiler in general:

| Operation | PB on 1.10 / PB on 1.13 | bare kernel on 1.10 / on 1.13 |
|:--|:--|:--|
| Weighted NormL1 value | 1.8–7.4× (Float64), 2.1–14× (Float32) | 1.0–1.4× (Float64) |
| `value_and_prox!`, weighted NormL1 | 1.3–4.0× (Float64), 1.4–6.7× (Float32) | 1.0–1.4× |
| NormL1 prox with a step per entry | 1.3–8.1× on Zen 3, 3.1–3.6× on Broadwell at 16,384 entries, 1.5–1.7× on Skylake-SP | 0.9–1.5× |
| NormL2 value | 1.3–2.3× | 1.0–1.2× |

- **The weighted NormL1 value loop is not vectorized on Julia 1.10.** That loop accumulates a floating-point sum
  and, in the same `@simd` loop, a Bool range flag built from two comparisons; LLVM 15 leaves it scalar and
  LLVM 20 vectorizes it. This is a property of that loop, not of Bool flags in general: the third review
  inspected the LLVM of the step-array passes of #925 and of candidate B925, which accumulate such a flag
  without a sum, and found them vectorized on both versions. An integer minimum over the coefficients' bit
  patterns in place of the flag vectorizes the value loop on both (`probes/wval_probe2.jl`). Draft PR #155
  replaces the flag with an integer reduction for the sake of Julia 1.14; §8.5 measures its port on both
  versions.
- **The scalar soft threshold has two data-dependent branches on Julia 1.10** (the sign tests inside `max` and
  `min`; §8.8). That is the step-array row of the table, and it is why that row depends on the CPU and on the
  length: a predictor that has learned a repeated input hides the branches. On inputs that change between
  calls, master's scalar loops on Julia 1.10 are 4 to 4.7 times slower on Zen 3 than a repeated-input benchmark
  reports.
- On Zen 3, small groups are also 1.3–1.6× slower on Julia 1.10 in PB and in the bare kernel alike: LLVM 15 has
  no vector epilogue, so groups below the unrolled vector length are summed by a scalar chain.
- On Skylake-SP every soft-thresholding loop is 1.4–3.2× slower on Julia 1.10, PB and bare kernel alike
  (512-bit code generation for `max`/`min`).
- No operation's *ranking* among implementations changes between the versions except the weighted value, where
  ProximalOperators (allocating, 3.5–6× slower than PB on 1.13) is only 1.2–1.5× slower on 1.10.

## 6. Numerical behavior

`bench/numerics.jl` evaluates every implementation on ordinary and exceptional inputs against 1024-bit BigFloat
references (`notes/numerics_julia1.13.md`, `notes/numerics_julia1.10.md`; the two files agree for every PB
tree).

- **PB master and every candidate tree** return the correctly rounded or few-eps result in every row: zero and
  -0 inputs, subnormals, entries near `sqrt(floatmax)` and `floatmax/4`, norms that overflow, squares that
  underflow, NaN and Inf entries, NaN and ±0 thresholds.
- **Plain sum of squares** (the level-1 norm kernel, BLAS `dot`): `Inf` when the squares overflow, 0 for a
  subnormal vector whose exact norm is representable. The same failure appears in a direct shrinkage: components
  wrong by 10^15 eps, or a vector left unshrunk.
- **`norm`, BLAS `nrm2`, the three-accumulator pass:** correct class everywhere, 4 to 11 times slower than PB in
  cache (§4).
- **PO `NormL2` prox:** correct on these inputs (it uses `norm`); returns -0 for negative entries of a vector
  shrunk to zero, where PB returns +0.
- **Soft thresholding:** the three-way select form loses a NaN input and the `Inf - Inf` case; the copysign form
  returns -0; `v - clamp(v, -c, c)` loses a NaN threshold. PB, PO and the two forms of §7.1 are correct on the
  whole grid.
- **Least-squares proximal operator on ill-conditioned designs** (60 × 12, relative error against BigFloat):

  | | cond 10², γ = 10⁶ | cond 10⁶, γ = 10⁶ | cond 10¹⁴, γ = 10⁶ | Float32: cond 10⁵, γ = 10⁶ |
  |:--|--:|--:|--:|--:|
  | PB (thin SVD) | 5.7e-15 | 8.9e-14 | 2.9e-13 | 2.8e-05 |
  | Cholesky of `XᵀX + I/γ` | 1.1e-13 | 5.3e-11 | 6.4e-11 | 2.9e-02 |
  | PO `LeastSquares` | 1.1e-13 | 5.3e-11 | 6.4e-11 | 2.9e-02 |

  At γ = 1 all three are at a few eps. The Cholesky route loses two to three digits in Float64 at large steps and
  is unusable in Float32 there (3% error). The reviewer also showed that an unchecked `potrf!` returns a wrong
  answer silently when the rounded Gram matrix loses the shift (`X = fill(1e8, 2, 2)`, γ = 1).

## 7. Candidate changes

Every candidate is a local branch off master in its own worktree (`pbopt/<name>`, branch `exp/kernelopt-<name>`);
the four of §9 were later published as draft pull requests #966 to #969, the others were not pushed. "Bit-identical" means: PB's test suite, the differential harnesses of `benchmark/` in exact
mode against master, and the checks named in the entry.

A Codex reviewer (gpt-6.1-sol, high effort) reviewed the patches three times: the first versions
(`notes/REVIEW-patches.md`), the revisions (`notes/REVIEW-patches-round2.md`), and the rebase on the new master
with the two pieces that were new there (`notes/REVIEW-patches-round3.md`).

- **Numerics.** Every review confirmed the numerical argument of A3, V and D3, with its own differential checks
  (28,082 assertions per Julia version in the third).
- **Performance.** The first review rejected B, C, C2 and D for slowdowns on small inputs or on branches the
  patch does not speed up; the versions below are the revisions. The second applied the 5% rule literally to
  arrays of a few entries and rejected A3, V and D3 for the cells that §8.2 then measured on the cluster. Each
  of the three therefore needs the maintainer to accept the cells listed there.
- **After the three patch reviews**, the confirmation on the new master found a cost of A3 that none of them
  and none of the earlier campaigns had seen (§8.7). A fourth review checked that finding
  (`notes/REVIEW-placement-round4.md`). §9 ranks the patches with it.
- What each review found is listed with its patch.

### 7.1 A3: the ordinary ℓ1 threshold path (ready; not recommended under the current policy, §9)

`shrink` computes `max(v - c, 0) + min(v + c, 0)`. Julia's `max` and `min` order the signed zeros and return NaN
for a NaN in either operand, which costs 7 vector instructions per vector on Julia 1.13 and 11.5 on Julia 1.10
(`probes/st_codegen.jl`), where the hardware maximum and minimum are one instruction each.

A3 adds `_shrink_nonneg(v, c) = (v - min(max(v, -c), c)) + 0`, with the maximum and minimum written as
two-operand selects, and uses it in the ordinary branch of `_l1_prox_threshold`, where the threshold `γλ0·w` is a
product of nonnegative factors. That is 4 instructions per vector. `shrink` itself is unchanged.

- **Equivalence.** For Float16, Float32 and Float64 the result equals `shrink(v, c)` for every `v` and every `c`
  that is not negative: +0, -0, positive, +Inf and NaN. Verified over all 65536 × 33792 Float16 pairs and
  2×10⁸ random Float32 and Float64 pairs on both Julia versions (`probes/st_clampz.jl`; the reviewer repeated the
  Float16 sweep independently), and through the public `prox!` and `prox_block` on 4.5 million outputs at extreme
  parameters. The `+ 0` is there for one pair: `v = -0` at a zero threshold (a zero weight), where the difference
  alone gives -0 and `shrink` gives +0. No existing test pinned that case; A3 adds one.
- **Why not a simpler form.** `v - clamp(v, -c, c)` (3 instructions) returns 0 for a NaN threshold, so a weight
  array mutated to NaN would give silent zeros instead of NaN. A search over the 32 arrangements of the two
  selects found none that matches `shrink` on all non-negative thresholds without the added zero
  (`probes/st_search.jl`).
- **Outside the domain.** At a negative threshold the two functions differ (`_shrink_nonneg(0.5, -1) = 1.5`,
  `shrink(0.5, -1) = 1`). The ordinary branch reaches a negative threshold only if a weight array is made
  negative after construction, which is outside the valid parameters; a test documents the difference.
- **Open design question for the maintainer.** A3 keeps two soft-thresholding functions, which the reviewer
  counts against the "no duplicated mathematical implementations" rule. The alternative is to give `shrink`
  itself the new formula for the IEEE types: one implementation, at the price of changing `shrink` at negative
  thresholds, where its docstring already requires `c ≥ 0`.
- **Rejected alternative: A** (the same sum as `shrink` with select-based max and min; bit-identical for every
  pair, 5 instructions). It is 6–19% faster than master on Julia 1.13 but **2.4× slower on Julia 1.10 for
  16-entry Float32 arrays** (3.5 ns → 8.5 ns in three fresh processes on the workstation; 2.65× on Broadwell).
  The reviewer measured A 7–9% faster than A2 on a Raptor Lake performance core; on Zen 3 and Broadwell the order
  is the opposite and larger (§8).
- **ElasticNet.** ElasticNet reaches the new formula through `_l1_prox_threshold` (a scalar step, `prox_block`,
  and the fallback loop of a step array), with no ElasticNet source changed. I also tried the formula at
  ElasticNet's two direct `shrink` calls (the zero-λ1 line and the ordinary line of #925's loop) and withdrew it
  after the third review. The thresholds there are provably not negative and the outputs were bit-identical,
  but it gained nothing on #925's step-array loop (0.94–1.01× with weights; 1.03–1.04× for unweighted Float32
  at 1024 and 65,536 entries on Julia 1.13) and lost on small calls (1.65× at 16 Float32 entries with λ1 = 0 on
  Julia 1.10). Reviewer's measurements, workstation.
- **Costs found on the cluster.** (1) On Zen 3, when the output array starts 16 to 256 bytes after the input
  modulo 4096 on huge pages, which is where Julia's allocator often puts two large arrays of a power-of-two
  length, A3 is up to 1.3× (NormL1) and 1.6× (ElasticNet) slower than master instead of 0.67–0.84× (§8.7).
  (2) On Julia 1.10, lengths 1 to 24 are 1.05–1.38× in some cells (§8.2, §8.6). (3) On Julia 1.13, scalar
  `prox_block` loops of ElasticNet over 4 to 100 entries and NormL1 `prox!` of 1 and 4 entries are 1.04–1.19×
  in some cells (§8.6). (4) A repeated sparse input on Zen 3 with Julia 1.10 is 1.15–1.35× (§8.8).
- **A gain that the first campaigns understated.** On Julia 1.10 A3 removes the two data-dependent branches
  of the ordinary scalar soft threshold: 3 to 3.6 times faster on sparse iterates whose values change at
  random between calls (§8.8).
- **Size.** 2 files, +20 −1 lines of source (one function with three methods, one call site), plus tests.

### 7.2 V: unweighted NormL1 and NormL2 values of 16 to 1024 entries (propose)

The unweighted NormL1 value is `λ0·sum(abs, x)` and the NormL2 value uses `sum(abs2, x)`. Julia's pairwise
reduction sums blocks of up to 1024 entries in one SIMD loop each, after taking the first two entries
separately and through its generic reduction layers. For an array of at most 1024 entries the whole sum is
therefore already one loop, and a direct loop does the same arithmetic two to three times faster for a few
hundred entries.

V sends strided Float32/Float64 arrays of 16 to 1024 entries to one out-of-line SIMD loop and leaves every
other length and array type on master's code.

- **Not bit-identical inside the window.** The lane count of the loop differs from Base's, so the last bits can
  differ: at most a few eps on the harness inputs. Outside the window (fewer than 16 entries, more than 1024)
  the values are master's bits.
- **Same rounding class.** Because Base's sum of at most 1024 entries is also a single loop, the error growth
  with the length is the same as on master. This is the difference from the first version of this patch (C and
  C2, one loop up to 2^17 entries as in the least-squares route), where the reviewer measured 323 eps for
  `NormL1(1f0)(fill(0.1f0, 2^17))` against about 1 eps on master. C and C2 are 0.55–0.80× up to 2^17 entries
  (§8) and remain available if the maintainer accepts that accuracy trade as in #918.
- **Lower limit.** Below 16 entries Base's small-array paths are as fast or faster, and the reviewer's
  singleton case (+7%) is in that range. With the loop out of line, arrays below 16 entries run master's code
  behind one length comparison.
- **SqrNormL2 is not changed**: its value of fewer than 16 entries read 10–20% slower with the same change on
  the workstation, and the family belongs to #910.
- **Size.** 2 files, +27 −3.

### 7.3 D3: groups with a zero sum of squares (propose)

When the sum of squares of a group is exactly zero, `_l2_shrink!` scans the group a second time to tell a zero
group from one whose squares all underflowed. D3 skips the scan when the threshold decides: if
`c ≥ |G|·√(floatmin/eps)` (2^-485 per entry in Float64, 2^-51 in Float32), then `c²` is far above the largest
sum of squares that can round to zero, so the proximal point is the zero group whatever the entries are, which
is also what the cold path returns.

- **Equivalence.** Bit-identical outputs and return codes on PB's two ℓ2 differential harnesses in exact mode
  (368 and 5,245 probes) and on tests that compare with the cold route: zero groups, groups whose squares all
  underflow at ordinary and at tiny thresholds, both sides of the guard, mixed precision.
- **Two defects of the first version (D), found in review.** (1) In Float16 the old bound overflowed for groups
  above 65,520 entries; the shortcut is limited to Float32 and Float64. (2) The old guard squared the
  threshold, and for a tiny threshold that product underflows: 13–16× slower on an efficiency core for zero
  groups at level `floatmin`, 10% on a performance core. D3 compares `c` with a normal number instead.
- **Remaining cost.** For a zero group whose threshold is below the guard (`c < |G|·2^-485`), D3 costs one
  multiplication and one comparison before the scan that master does anyway: +7–15% on groups of 1 to 5 entries
  on the workstation (about 0.3 ns per group), nothing at 1024.
- **Size.** 1 file, +15 −1.

### 7.4 B2: NormL1 with a step per entry (a trade-off, for the maintainer to decide)

With a step array, the range test of `γᵢλ0` is a branch per entry and the generic loop stays scalar. B2 tests the
steps a block of 256 at a time (`_steps_normal`: integer minimum and maximum of the bit patterns, which vectorize
on both Julia versions, where the conjunction of two floating-point comparisons does not on 1.10). While a block
is ordinary it is done by a loop with the ordinary branch alone; from the first block that is not, the rest of
the array is done entry by entry as on master. Arrays below 8 entries and zero or non-finite levels skip the
blocks.

- **Equivalence.** Bit-identical to the generic loop on every tested input (5,832 assertions for the first
  version: levels of other types, zero and infinite levels, weights with 0 and Inf, in-place calls, one
  exceptional step at each position).
- **Gain.** Step arrays of 64 or more ordinary steps: 0.39–0.47× in Float64 and 0.14–0.30× in Float32 on the
  workstation; §8 has the cluster numbers.
- **Cost, measured on the workstation (Julia 1.13.4).** Fewer than 8 entries: +10–25% (0.5–1 ns). One
  exceptional step in an array of 8 to 64: +12–80%. All steps exceptional (all zero): +7–40%. On an efficiency
  core even 8 ordinary Float64 steps are 38% slower, because 64-bit integer minimum and maximum are emulated on
  AVX2.
- **Why it was not in the proposal set.** The maintainer had ruled against scans of step arrays that cost
  6–10% (#874, #909). B2 is faster by 2.5–7× where steps are ordinary and slower by 10–80% on short arrays and
  on arrays with exceptional steps.
- **Size.** 2 files, +50.

**B925, its successor on the new master.** #925 then settled the same question for ElasticNet: one pass over
the steps in blocks of 256, and a loop specialized to the ordinary branches when every step passes, the generic
loop otherwise. B925 is that design for NormL1, line for line (`_L1OrdinarySteps`, `_l1_ordinary_steps`); it
replaces B2, whose blocked tail and integer reductions are gone.

- **Equivalence.** Bit-identical to master on PB's NormL1 harnesses (56 and 144 probes) and on `ridge910.jl`
  (76), on both Julia versions; the tests compare with `prox_block` and with the generic loop for views,
  reshapes, abstract element types, sparse and block iterates. The reviewer found no deviation from #925 that
  matters, and no ambiguity.
- **Third review: approve with changes.** Its measurements (workstation, performance core, B925 against master
  beside #925 against the commit before it, on the same inputs):

  | | NormL1, B925 / master | ElasticNet, #925 / before |
  |:--|:--|:--|
  | ordinary steps, 1024 entries | 0.15–0.33 | 0.19–0.57 |
  | one exceptional step at the end, 1024 entries | 1.05–1.13 | 0.97–1.17 |
  | every step exceptional, 1024 entries | 1.01–1.11 | 0.99–1.14 |
  | 1 entry, ordinary (Float32, Julia 1.13) | 1.67 | 1.86 |
  | 8 entries, ordinary (Float32, Julia 1.13) | 1.05 | 0.58 |
  | 8 entries, last step zero (Julia 1.13) | 1.60 (+5.7 ns) | 1.32 (+6.5–6.7 ns) |
  | first call, extra compile time | 26–59 ms | 21–73 ms |

  At 1024 entries B925 asks for what #925 was given. At a few entries its absolute cost is the same or smaller
  and its relative cost larger, because the NormL1 loop it replaces is cheaper. The reviewer's change is to
  skip the pass for short arrays; that is not implemented.
- **Size.** 1 file, +34 lines of source, plus tests.

### 7.5 Not proposed

| Candidate | What it does | Outcome |
|:--|:--|:--|
| A | select-based `max`/`min` inside `shrink` | **Rejected**: 2.4× slower on Julia 1.10 at 16 Float32 entries. |
| C, C2 | one-loop value sums up to 2^17 entries (NormL1; NormL2 and SqrNormL2) | **Superseded by V**: hundreds of eps at 2^17 Float32 entries; small-array slowdowns. |
| B, D | first versions of B2 and D3 | **Superseded**: the review's findings above. |
| D2 | the zero test fused into the sum-of-squares loop (an OR of bit patterns), which would remove the second pass in values too | **Rejected**: dense-group values up to 1.3× slower on Julia 1.13 and 1.9× on Julia 1.10 (workstation), for 0.6× on sparse values. |
| F | `@inline` on the compact `prox_block!` of GroupNormL2 | **Not proposed**: 0.65–0.85× on a sweep of small blocks, nothing in block coordinate descent (§8), larger call sites. |
| Fused NormL1 `value_and_prox!` | prox and value in one pass | **Not built**: 0.46–0.60× at kernel level, but it changes the summation order of the fused value and duplicates the value's range checks. |
| Cached Cholesky, or a cached `V·diag·Vᵀ`, for the least-squares prox | fewer products per call at a fixed step | **Rejected**: §6 (accuracy) and §4 (PB's two `gemv` are already 1.4–2.8× faster per call than `potrs`). |
| Fixed-size code for groups of 2 to 10 entries | remove the vector preamble | **Not built**: the gap to the floor is the range checks, not the loop (§7.6). |

### 7.6 Where the small-group gap comes from

Hardware counters on the workstation (`probes/perf_group.jl`, Julia 1.13.4) for groups of 2 to 32 entries:

| Operation | PB instructions per group | check-free kernel | extra branches | mispredictions |
|:--|--:|--:|--:|--:|
| `prox!`, m = 2 | 86 | 61 | 5 | none added |
| `value_and_prox!`, m = 2 | 105 | 65 | 8 | none added |
| value, m = 2 | 66 | 39 | 6 | none added |
| value, m = 32 | 96 | 68 | 7 | none added |

The difference is constant per group: about 25 to 45 instructions and 5 to 9 well-predicted branches, the range
tests that let the ordinary path be trusted (zero weight, the range of `λw`, a finite and normal sum of squares,
a value below `floatmax/2`). It is 20–45% of a group of 2 to 10 entries and under 3% at 128. Removing it means
removing guarantees, so it is not a candidate.

## 8. Results for the candidates

### 8.1 Tree comparison at 16 entries and above (`formal4`, baseline `50db4b1e`)

Six fresh processes per stratum, 21 rounds each, 540 cases (ℓ1, ℓ2 and group cases of 16 to 2^20 entries, both
element types, plus the solver loops). A cell is the geometric mean over the cases of the per-process median
ratio to master, with the smallest and largest per-process ratio of any case in brackets.

| tree | cases (count) | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Broadwell, 1.10 | Skylake-SP, 1.13 |
|:--|:--|:--|:--|:--|:--|:--|
| A3 | l1/prox/Float64 (6) | 0.74 [0.67–1.00] | 0.65 [0.54–0.89] | 0.76 [0.64–0.98] | 0.74 [0.60–0.91] | 0.84 [0.74–1.06] |
| A3 | l1/prox_w/Float64 (6) | 0.84 [0.68–1.01] | 0.71 [0.52–0.99] | 0.81 [0.67–1.00] | 0.76 [0.61–0.98] | 0.90 [0.82–1.02] |
| A3 | l1/valprox/Float64 (6) | 0.81 [0.74–0.97] | 0.74 [0.65–0.93] | 0.81 [0.73–1.00] | 0.80 [0.68–0.93] | 0.89 [0.81–1.09] |
| A3 | l1/prox/Float32 (6) | 0.76 [0.67–1.04] | 0.65 [0.52–0.96] | 0.75 [0.64–1.00] | 0.70 [0.55–1.00] | 0.84 [0.75–1.15] |
| A3 | l1/prox_w/Float32 (6) | 0.83 [0.59–1.00] | 0.72 [0.55–0.97] | 0.78 [0.66–1.00] | 0.72 [0.59–1.00] | 0.91 [0.82–1.00] |
| A3 | l1/valprox/Float32 (6) | 0.82 [0.74–1.02] | 0.74 [0.63–0.98] | 0.80 [0.72–1.00] | 0.75 [0.65–0.94] | 0.90 [0.83–1.02] |
| B2 | l1/prox_g/Float64 (6) | 0.35 [0.28–0.69] | 0.21 [0.04–0.61] | 0.41 [0.33–0.88] | 0.27 [0.12–0.73] | 0.36 [0.22–0.75] |
| B2 | l1/prox_gw/Float64 (6) | 0.35 [0.27–0.69] | 0.17 [0.04–0.55] | 0.47 [0.33–0.90] | 0.31 [0.14–0.74] | 0.39 [0.23–0.85] |
| B2 | l1/prox_g/Float32 (6) | 0.18 [0.13–0.59] | 0.09 [0.02–0.65] | 0.18 [0.13–0.49] | 0.12 [0.05–0.60] | 0.22 [0.14–0.61] |
| B2 | l1/prox_gw/Float32 (6) | 0.18 [0.11–0.59] | 0.11 [0.02–0.64] | 0.19 [0.13–0.56] | 0.12 [0.05–0.60] | 0.21 [0.11–0.57] |
| D3 | l2/prox/Float64 (14) | 0.92 [0.70–1.04] | 0.91 [0.61–1.04] | 0.92 [0.71–1.07] | 0.90 [0.69–1.06] | 0.92 [0.71–1.13] |
| D3 | group/prox/Float64 (12) | 0.85 [0.74–1.00] | 0.85 [0.74–0.96] | 0.90 [0.81–0.99] | 0.86 [0.75–0.94] | 0.86 [0.74–1.00] |
| D3 | group/valprox/Float64 (12) | 0.86 [0.74–0.98] | 0.84 [0.69–1.01] | 0.90 [0.79–1.01] | 0.87 [0.72–1.00] | 0.86 [0.74–0.99] |
| D3 | group/block_sweep/Float64 (6) | 0.90 [0.81–0.98] | 0.91 [0.83–1.00] | 0.92 [0.87–0.98] | 0.91 [0.87–1.00] | 0.91 [0.78–1.02] |
| D3 | l2/prox/Float32 (14) | 0.94 [0.69–3.42] | 0.91 [0.63–1.12] | 0.93 [0.65–1.09] | 0.90 [0.76–1.02] | 0.90 [0.70–1.05] |
| D3 | group/prox/Float32 (12) | 0.86 [0.73–0.98] | 0.82 [0.70–0.95] | 0.85 [0.75–0.97] | 0.84 [0.70–0.93] | 0.84 [0.73–0.98] |
| V | l1/value/Float64 (6) | 0.83 [0.61–1.00] | 0.75 [0.52–1.00] | 0.79 [0.45–1.00] | 0.70 [0.52–1.00] | 0.87 [0.68–1.00] |
| V | l2/value/Float64 (14) | 0.86 [0.49–1.15] | 0.82 [0.42–1.19] | 0.84 [0.50–1.08] | 0.78 [0.28–1.20] | 0.88 [0.48–1.18] |
| V | l1/value/Float32 (6) | 0.78 [0.49–1.00] | 0.64 [0.31–1.18] | 0.75 [0.51–1.00] | 0.64 [0.30–1.00] | 0.80 [0.50–1.00] |
| V | l2/value/Float32 (14) | 0.82 [0.29–1.14] | 0.71 [0.32–1.10] | 0.80 [0.37–1.07] | 0.64 [0.18–1.16] | 0.85 [0.53–1.17] |

Decisions by the rule of §2.2 (a case is slower only if it is above 1.05 in every process and the A/A control
stays within ±2%; faster likewise below 0.95, so the "faster" counts are conservative: B2's step-array cases,
whose master loop has a noisy control, are mostly not counted):

| tree | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Broadwell, 1.10 | Skylake-SP, 1.13 |
|:--|:--|:--|:--|:--|:--|
| A/A control | 0 slower, 0 faster (0 unresolved) | 0 slower, 0 faster (0 unresolved) | 0 slower, 0 faster (0 unresolved) | 0 slower, 0 faster (0 unresolved) | 0 slower, 0 faster (6 unresolved) |
| A3 | 0 slower, 29 faster (0 unresolved) | 0 slower, 28 faster (0 unresolved) | 0 slower, 51 faster (0 unresolved) | 0 slower, 48 faster (0 unresolved) | 0 slower, 22 faster (0 unresolved) |
| B2 | 0 slower, 6 faster (0 unresolved) | 0 slower, 3 faster (0 unresolved) | 0 slower, 24 faster (0 unresolved) | 0 slower, 16 faster (0 unresolved) | 0 slower, 0 faster (0 unresolved) |
| D3 | 0 slower, 19 faster (0 unresolved) | 0 slower, 22 faster (0 unresolved) | 0 slower, 41 faster (0 unresolved) | 0 slower, 37 faster (0 unresolved) | 0 slower, 35 faster (0 unresolved) |
| FINAL2 | 0 slower, 63 faster (0 unresolved) | 0 slower, 69 faster (0 unresolved) | 0 slower, 110 faster (0 unresolved) | 0 slower, 103 faster (1 unresolved) | 0 slower, 71 faster (0 unresolved) |
| V | 0 slower, 15 faster (0 unresolved) | 0 slower, 18 faster (1 unresolved) | 0 slower, 18 faster (0 unresolved) | 0 slower, 17 faster (0 unresolved) | 0 slower, 15 faster (0 unresolved) |

**No tree has a reproducibly slower case in any stratum**, and the A/A control has none either. FINAL2 is A3, V
and D3 together.

### 8.2 Tiny inputs and failing guards (`tiny1`, baseline `50db4b1e`)

The patch review found slowdowns that the cases above cannot show: arrays of a few entries, and zero groups at a
threshold below D3's guard. `tiny1` measures them: 1 to 100 entries for every ℓ1 and ℓ2 operation and zero groups
at level `floatmin`, six processes on Zen 3 and on Broadwell for each Julia version (360 cells for A3 and V, 304
for D3). Cells that are above 1.05 in every process:

| Patch | Julia 1.13.4 | Julia 1.10.13 |
|:--|:--|:--|
| A3 | Zen 3: none. Broadwell: a 1-entry array, 1.10–1.13× (0.3 ns) | Float64, unweighted: 8 entries 1.13× (Zen 3) and 1.29× (Broadwell), 9 entries 1.14×, 24 entries 1.09×. Float64, weights: 12 entries 1.16×, 4 entries 1.06×. Float32, weights: 8 entries 1.23–1.32×, 9 entries 1.21×, 24 entries 1.34–1.38× |
| V | none | NormL2 value, Float32: 1 entry 1.11–1.12×, 2 entries 1.09× |
| D3 | Zen 3: none. Broadwell: zero groups of 1 and 5 entries at level `floatmin`, 1.10–1.15× | none |

Every other cell is equal or faster; `notes/tables/tiny1-{A3,V,D3,FINAL2}.md` has all of them.

- **A3's cost is a Julia 1.10 effect at particular lengths.** LLVM 15 unrolls the cheaper loop body four times,
  so the vector code needs 16 (Float64) or 32 (Float32) entries where master's heavier body needed 8; lengths
  between the two fall to the scalar remainder. Lengths 1 to 7 are faster than master (0.65–0.96×), and so are
  lengths from 15 (Float64). On Julia 1.13 there is a vector epilogue and no such gap. The absolute cost is 1 to
  2.5 ns per call.
- **V and D3 cost a fraction of a nanosecond** in the cells listed, and only there.

### 8.3 Solver loops

On Zen 3 every proximal gradient, accelerated proximal gradient, coordinate descent and block coordinate descent
loop is within 0.99–1.00 of master for A3, V, D3 and FINAL2, on both Julia versions and both element types, for
dense designs from 200 × 200 to 100 × 10000 and sparse designs up to 2000 × 100000. The gradient is one to two
matrix products; the proximal step is about 1% of an iteration.

The exception is proximal gradient in a diagonal metric (a step per entry): B2 is 0.96–0.98 on Julia 1.13 and
0.86–0.92 on Julia 1.10, and A3 alone is 0.90–0.93 on Julia 1.10 there (its scalar form in the unvectorized
loop).

**The kernel gains do not reach solver iterations of regression workloads.** They matter where the proximal
step is the iteration: separable problems, splitting methods without a matrix product, repeated thresholding of
long vectors.

### 8.4 ProximalBase's own harnesses (`native3`, FINAL2 against `50db4b1e`, Broadwell)

`benchmark/norml1_infinite_weight.jl` and `benchmark/norml1_value_underflow.jl`, 21 rounds, in both role orders;
a case counts when it is beyond ±5% in both orders. These harnesses cover the storage types my cases do not
(sparse and block iterates, aliased outputs, infinite and zero weights).

| Harness | Julia | cases | faster in both orders | slower in both orders |
|:--|:--|--:|--:|--:|
| `norml1_infinite_weight` | 1.13.4 | 56 | 34 | 0 |
| `norml1_infinite_weight` | 1.10.13 | 56 | 45 | 0 |
| `norml1_value_underflow` | 1.13.4 | 136 | 8 | 0 |
| `norml1_value_underflow` | 1.10.13 | 136 | 23 | 0 |
| `l2_norms` | 1.13.4 | 368 | 49 | 3, none with a clean control |
| `l2_norms` | 1.10.13 | 368 | 52 | 3, none with a clean control |

The value harness was run without its dense unweighted probes, whose last bits V changes (its check stops at
the first output difference). One forward-order reading of 1.165 (weighted block iterate, step array, 1024
Float32 entries) is 0.950 in the reversed order, with an A/A control between 0.51 and 1.17 on those cells: code
placement, as §10 item 1 explains for that harness.

The six `l2_norms` cases that are beyond 5% in both orders (1.05–1.33) are four values (three of groups of 10
entries on Julia 1.10, one sparse-matrix NormL2 value on Julia 1.13), on paths that no patch changes, and two
`value_and_prox!` cells of groups of 2 entries on Julia 1.13. In each of the six the A/A control of the same
run is off by 5 to 23% in at least one order (for example 1.063 and 1.330 with controls 1.093 and 0.766), so
none of them is a resolved slowdown. They are the dynamic-dispatch probes of §10 item 1.

### 8.5 The port of draft PR #155 (`formal2`, six processes)

| | Broadwell, 1.13 | Broadwell, 1.10 | Skylake-SP, 1.13 |
|:--|:--|:--|:--|
| weighted NormL1 value, Float64 | 0.80 [0.70–0.99] | 0.22 [0.17–0.47] | **1.16 [1.01–1.28]** |
| weighted NormL1 value, Float32 | 0.70 [0.56–0.99] | 0.11 [0.07–0.44] | 0.97 [0.90–1.10] |
| `value_and_prox!`, weights, Float64 | 0.92 [0.87–0.99] | 0.45 [0.36–0.68] | **1.06 [1.02–1.10]** |
| cases slower in every process | 0 | 0 | **7** |

#155 makes the weighted value 4.5 to 9 times faster on Julia 1.10, which its description does not mention, and
0.70–0.80× on Julia 1.13 on AVX2 machines. On the AVX-512 node with Julia 1.13.4 it is slower than master in all
six processes: 1.14–1.28× for the Float64 value at 128 to 16,384 entries and 1.05–1.10× for its
`value_and_prox!`. The PR's own measurements were on an EPYC 7513 only.

### 8.6 Confirmation on the new master (`18b27149`, after #909 and #925)

The trees were rebased on the new master; V, D3 and B925 are unchanged by the rebase, and A3 and FINAL
(= A3 + V + D3) have master's `elasticnet.jl`. Sources: `formal6` for every tree on Zen 3 and for A3 and FINAL
everywhere, `formal5` for V, D3 and B925 on the Intel nodes. 606 cases: those of §8.1 plus ElasticNet. Six fresh
processes per stratum. Float64 rows; the Float32 rows are in `notes/tables/report-formal6.md`.

| tree | cases (count) | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Broadwell, 1.10 | Skylake-SP, 1.13 |
|:--|:--|:--|:--|:--|:--|:--|
| A3 | l1/prox/Float64 (6) | 0.73 [0.67–1.00] | 0.65 [0.56–0.89] | 0.77 [0.65–0.99] | 0.75 [0.59–0.91] | 0.84 [0.74–1.02] |
| A3 | l1/prox_w/Float64 (6) | 0.84 [0.69–1.00] | 0.71 [0.54–0.98] | 0.81 [0.67–1.00] | 0.76 [0.61–0.99] | 0.90 [0.80–1.00] |
| A3 | l1/valprox/Float64 (6) | 0.81 [0.74–0.96] | 0.74 [0.62–0.92] | 0.82 [0.72–1.01] | 0.80 [0.66–0.93] | 0.89 [0.79–1.01] |
| A3 | l1/valprox_w/Float64 (6) | 0.92 [0.83–1.00] | 0.94 [0.88–1.00] | 0.90 [0.81–1.00] | 0.94 [0.84–0.99] | 0.94 [0.88–1.02] |
| A3 | enet/prox/Float64 (5) | 0.86 [0.68–1.43] | 0.66 [0.53–1.04] | 0.80 [0.68–0.99] | 0.68 [0.49–0.92] | 0.90 [0.79–1.06] |
| A3 | enet/prox_w/Float64 (5) | 0.93 [0.77–1.17] | 0.75 [0.56–0.97] | 0.86 [0.68–1.00] | 0.78 [0.59–0.98] | 0.94 [0.88–1.01] |
| A3 | enet/prox_g/Float64 (5) | 1.00 [0.97–1.04] | 1.00 [1.00–1.04] | 1.00 [1.00–1.01] | 1.00 [1.00–1.00] | 1.00 [0.97–1.01] |
| A3 | enet/valprox/Float64 (5) | 0.91 [0.77–1.32] | 0.75 [0.63–1.03] | 0.85 [0.74–0.99] | 0.75 [0.55–0.93] | 0.92 [0.86–1.01] |
| A3 | enet/block_loop/Float64 (4) | 0.93 [0.87–1.02] | 0.72 [0.59–1.00] | 0.94 [0.77–1.10] | 0.78 [0.64–1.66] | 0.95 [0.89–1.10] |
| B925 | l1/prox_g/Float64 (6) | 0.36 [0.28–0.72] | 0.21 [0.04–0.65] | 0.39 [0.30–0.61] | 0.25 [0.12–0.57] | 0.40 [0.26–0.95] |
| B925 | l1/prox_gw/Float64 (6) | 0.37 [0.26–0.72] | 0.18 [0.05–0.69] | 0.45 [0.31–0.93] | 0.30 [0.14–0.63] | 0.39 [0.25–1.03] |
| D3 | l2/prox/Float64 (14) | 0.91 [0.70–1.06] | 0.89 [0.60–1.14] | 0.91 [0.71–1.07] | 0.89 [0.70–1.08] | 0.90 [0.37–1.15] |
| D3 | l2/valprox/Float64 (14) | 0.91 [0.69–1.05] | 0.90 [0.67–1.02] | 0.91 [0.71–1.04] | 0.90 [0.71–1.08] | 0.95 [0.71–1.39] |
| D3 | group/prox/Float64 (12) | 0.85 [0.76–0.97] | 0.83 [0.71–0.96] | 0.89 [0.74–1.01] | 0.86 [0.68–1.21] | 0.86 [0.74–1.00] |
| D3 | group/valprox/Float64 (12) | 0.86 [0.75–0.98] | 0.85 [0.74–0.99] | 0.90 [0.80–1.01] | 0.88 [0.80–0.94] | 0.85 [0.73–0.98] |
| D3 | group/block_sweep/Float64 (6) | 0.90 [0.82–0.98] | 0.91 [0.84–0.99] | 0.92 [0.87–0.99] | 0.91 [0.86–1.01] | 0.91 [0.86–0.97] |
| V | l1/value/Float64 (6) | 0.83 [0.60–1.00] | 0.75 [0.48–1.00] | 0.79 [0.42–1.00] | 0.69 [0.36–1.00] | 0.86 [0.68–1.01] |
| V | l2/value/Float64 (14) | 0.87 [0.50–1.14] | 0.82 [0.43–1.19] | 0.84 [0.39–1.11] | 0.78 [0.39–1.20] | 0.87 [0.53–1.17] |

ElasticNet has no source change in A3: it reaches the new formula through `_l1_prox_threshold`. Its step-array
loop (`prox_g`), which #925 specialized with its own `shrink` call, is unchanged at 1.00.

Decisions by the rule of §2.2:

| tree | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Broadwell, 1.10 | Skylake-SP, 1.13 |
|:--|:--|:--|:--|:--|:--|
| A/A control | 0 slower, 0 faster | 0 slower, 0 faster | 0 slower, 0 faster | 0 slower, 0 faster | 0 slower, 0 faster |
| A3 | 0 slower, 37 faster | **2 slower**, 55 faster | 0 slower, 71 faster | 0 slower, 84 faster | 0 slower, 40 faster |
| V | 0 slower, 16 faster | 0 slower, 18 faster | 0 slower, 18 faster | 0 slower, 17 faster (3 unresolved) | 0 slower, 11 faster |
| D3 | 0 slower, 18 faster | 0 slower, 28 faster | 0 slower, 37 faster | 0 slower, 48 faster | 0 slower, 30 faster |
| B925 | 0 slower, 4 faster | 0 slower, 8 faster | 0 slower, 25 faster | 0 slower, 18 faster | 0 slower, 1 faster |
| FINAL | 0 slower, 71 faster | **3 slower**, 100 faster | 0 slower, 129 faster | 0 slower, 138 faster | 0 slower, 81 faster |

- **A3 has two slower cases, on Zen 3 with Julia 1.10** (node b09-17, all six processes, control within ±1%):
  ElasticNet `prox!` at 2^20 Float32 entries, 1.20–1.29× (165 µs → 199–213 µs), and its `value_and_prox!`,
  1.14–1.23×. The NormL1 `prox!` on the same input in the same processes is 0.73× (125 µs → 91 µs). FINAL has
  the same two and V's two-entry Float32 NormL2 value (1.10×, §8.2).
- **The same million-entry cases are unstable on Zen 3 with Julia 1.13.4.** In the six processes of `formal6`
  (node a03-04) ElasticNet at 2^20 Float32 entries reads 91, 139, 141, 141, 161 and 162 µs with A3 against
  110–129 µs on master, and NormL1 reads 90–91 µs in four processes and 131 and 220 µs in two. A second node
  (b04-18, `big`) gives 91, 91, 118, 121, 125 and 143 µs against 110–112 µs, and with Julia 1.10 it gives 102 µs
  in two processes and 155–161 µs in four against 153–155 µs. Master does not move between processes on that node;
  A3 does. At 16,384 entries and below every process on every node agrees (for example 1.19 µs against 1.70 µs).
  §8.7 finds the cause: the placement of the output array relative to the input.
- **One more cell to state** (Broadwell, Julia 1.13.4): the scalar `prox_block` loop of ElasticNet over 16
  Float64 entries is 1.04–1.10× with A3 (five of six processes above 1.05; 18.0 ns → about 19.3 ns for the 16
  calls). The same cell is 0.90–0.92 in Float32, and 0.88–0.93 on Julia 1.10 in five processes.
- **NormL1 with a step array and weights, Zen 3, Julia 1.13.4:** A3 reads 0.96 in some processes and 1.10 in
  others (Float32 median 1.10, Float64 1.00), and so does master's own second compilation: 1,275 ns or 1,409 ns
  for 1024 entries, for master, its control, V and D3 alike. It is a two-level code-placement effect of that
  scalar loop, not a property of a tree, and the rule does not call it. B925 replaces the loop (190–380 ns).

**Tiny inputs (`tiny3`):** the cases of §8.2 plus ElasticNet, 706 cells, six processes per stratum on Zen 3 and
on Broadwell with both Julia versions. Cells above 1.05 in every process with a clean control, among the cells
that run code the patch changes:

| Patch | Zen 3, 1.13 | Zen 3, 1.10 | Broadwell, 1.13 | Broadwell, 1.10 |
|:--|:--|:--|:--|:--|
| A3 (390 cells) | 2: scalar `prox_block` loop of ElasticNet, Float32, 4 entries 1.11–1.14× and 100 entries 1.06–1.12× | 7: lengths 8, 9, 12 and 24, 1.06–1.36× (§8.2) | 4: the same loop at 4 entries 1.06–1.10× and 12 entries 1.17–1.19×; NormL1 `prox!`, Float32, 1 entry 1.12× and 4 entries 1.07× | 14: lengths 1 to 24, 1.05–1.34× |
| V (270) | none | none | NormL2 value, Float64, 2 entries: 1.06–1.08× | NormL2 value, Float32, 1 entry: 1.08–1.13× |
| D3 (256) | none | none | zero groups of 5 entries at level `floatmin`: 1.10–1.11× | NormL2 `prox!`, Float32, 3 entries: 1.06–1.08× |
| B925 (60) | 17: 1 to 9 entries, 1.06–2.14× | 6: 1 to 7 entries, 1.30–2.55× | 11: 3 to 7 entries, 1.15–1.85× | 10: 1 to 7 entries, 1.37–2.20× |

The same rule also calls 7 cells slower that the tree cannot affect, out of about 7,400 such cells: six ElasticNet
cells for D3 on Broadwell with Julia 1.10 (1.06–1.10×) and one NormL2 cell for A3: 0.1% of those cells,
against 1.7% of A3's own. That is a description, not a calibrated error rate (the cells share code and
layout), and it does not excuse a cell of a patch's own code. Faster in every process: A3 15 to
108 cells per stratum, V 18 to 30, D3 2 to 23. Lists: `notes/tables/tiny3-slowcells.txt` and
`notes/tables/tiny3-{A3,V,D3,B925}.md`.

**PB's own harnesses on the new master (`native5`, FINAL, A3 and B925 as roles, both orders):**

| Harness | CPU, Julia | cases | faster in both orders | slower in both orders |
|:--|:--|--:|--:|:--|
| `norml1_infinite_weight` | Broadwell, 1.13.4 | 56 | 33 | 0 |
| `norml1_infinite_weight` | Broadwell, 1.10.13 | 56 | 45 | 0 |
| `norml1_value_underflow` | Broadwell, 1.13.4 | 136 | 9 | 3, none with a clean control |
| `norml1_value_underflow` | Broadwell, 1.10.13 | 136 | 23 | 0 |
| `l2_norms` (the combined tree of `m2`; its V and D3 code is final) | Broadwell, 1.13.4 | 368 | 50 | 0 |
| `l2_norms` (the same) | Broadwell, 1.10.13 | 368 | 53 | 1, see below |
| `ridge910` (FINAL / A3) | Broadwell, 1.13.4 | 76 | 10 / 13 | 0 / 0 |
| `ridge910` (FINAL / A3) | Broadwell, 1.10.13 | 76 | 16 / 16 | 0 / 0 |
| `ridge910` (FINAL / A3) | Zen 3, 1.13.4 | 76 | 4 / 4 | 3 / 1, see below |
| `ridge910` (FINAL / A3) | Zen 3, 1.10.13 | 76 | 16 / 17 | 2 / 2, see below |

- The three `norml1_value_underflow` cells (1.05–1.08×) have A/A controls of 1.06–1.08 in one order and 0.93–1.00
  in the other.
- The one `l2_norms` cell is the NormL2 value of a Float32 sparse matrix (44 ns): 1.25× and 1.30× in the two
  orders, with A/A controls of 0.99 and 0.77. No patch changes that path (V takes strided dense arrays only),
  and the probe is one of the dynamic-dispatch probes of §10 item 1; a cell of the same kind appeared on the
  old baseline (§8.4). I count it as unresolved, and it is the one harness cell with a clean control in one
  order that I cannot explain.
- **`ridge910`, Zen 3, Julia 1.13.4.** FINAL reads 1.05–1.11× on the NormL1 step-array loop (the two-level
  effect above: A3, which has the same code there, reads 0.96 and 1.05–1.11 in the two orders) and 1.05× on a
  `prox_block` loop of the centred ridge, whose control reads 1.05 in one order. A3's one cell is the ridge
  value at 1024 Float32 entries (1.08×, 44 ns), which no patch touches.
- **`ridge910`, Zen 3, Julia 1.10.** `enet/sparse_prox!/scalar` (a sparse iterate with 6,553 stored entries) is
  1.15–1.22× with A3 in both orders with a clean control (10.2 µs → 12 µs). On Broadwell the same case is
  0.43–0.51×. §8.8 explains both: the benchmark repeats one input, and on Julia 1.10 master's loop has
  data-dependent branches that the Zen 3 predictor learns.
- B925 has one cell in both orders on Broadwell with Julia 1.10 (a `prox_block` loop of the centred ridge with
  a step array, 1.76–1.85×, on a path B925 does not change; the control of the neighbouring cell of the same
  probe reads 1.08 and 1.21 in the two orders) and one on Zen 3 with Julia 1.10 (ElasticNet, mixed step array,
  1024 Float32 entries, 1.06–1.07×, also not its code). I count both as unresolved.

### 8.7 Array placement: the output shortly after the input modulo 4096, on huge pages

`bench/layout_probe.jl` and `bench/offset_probe.jl` time master, A3 and master's second compilation on the same
two arrays, which the probes place themselves: true `Array`s wrapped around 2 MiB-aligned regions, on transparent
huge pages or on 4 KiB pages (`madvise` before the first touch), with the output shifted against the input by a
chosen number of bytes. Two to three fresh processes per node and Julia version, 11 or 15 alternating rounds.
A fourth Codex review (`notes/REVIEW-placement-round4.md`) checked the probes, recomputed the tables from the
CSVs and corrected several statements of the first version of this section; its corrections are in the text
below and its limits are listed at the end.

1. **Page backing matters inside the band, not outside it.** At offsets 0 and 2048 A3/master at 2^20 entries
   is the same on regions advised to huge pages, on 4 KiB pages and with one array on each: 0.82–0.85 on Zen 3
   with Julia 1.13.4 and 0.65–0.78 with Julia 1.10, in every process on both nodes. At +64 and +128 bytes it is
   not: on huge-page regions A3 is 1.3–1.8× its own time at offset 0 and 1.00–1.45× master (Julia 1.13.4, Zen 3,
   `probe5`); on 4 KiB pages it is 1.02–1.05× its own time and 0.86–0.87× master (0.83–0.99 in `probe1`). The
   same holds on Julia 1.10 (0.65–0.93 on 4 KiB pages against up to 1.38 on huge pages) and on the Intel nodes.
   The two backings are separate regions, so this does not isolate the page size from the physical placement
   that comes with it, and `madvise` is a request: `layout_probe.jl` records the huge-page coverage of each
   mapping, `offset_probe.jl` does not. Everything below is on huge-page regions. The cluster nodes run with
   transparent huge pages set to `always`, and the mappings of the arrays that Julia allocated itself in
   `probe1` were at least partly covered by huge pages.
2. **The offset `(pointer(y) - pointer(x)) mod 4096` matters when it is between 16 and about 256 bytes.** That
   band is a description of where most of the penalty is, not a sharp boundary: the profile is serrated (at
   1024 entries +64 often costs nothing while +16, +48 and +80 do), on Zen 3 at 16,384 entries A3 is still 1.12×
   its own time at +512, and on Skylake-SP the tail reaches +384 and +512 (1.2–1.3×). In the band most loops
   are slower than the same loop at offset 0, on every CPU; the exceptions are in the table (the ridge loop
   and the broadcast at 2^20 entries on Broadwell, `copyto!` at some sizes). Largest slowdown of a loop against itself at offset 0 (`probe5`, Julia 1.13.4, over the offsets
   16 to 256 and both element types), at 2^20 / 16,384 / 1024 entries:

   | | Zen 3 | Broadwell | Skylake-SP |
   |:--|:--|:--|:--|
   | master, NormL1 `prox!` | 1.50 / 1.38 / 1.27 | 1.48 / 1.80 / 1.77 | 1.43 / 3.14 / 3.56 |
   | A3, NormL1 `prox!` | 1.62 / 2.02 / 1.44 | 1.30 / 2.23 / 2.25 | 1.34 / 3.84 / 4.56 |
   | master, ElasticNet `prox!` | 1.52 / 1.54 / 1.41 | 1.58 / 1.92 / 1.86 | 1.43 / 2.90 / 3.26 |
   | A3, ElasticNet `prox!` | 1.85 / 2.17 / 1.61 | 1.42 / 2.47 / 2.33 | 1.44 / 3.58 / 3.93 |
   | master, SqrNormL2 `prox!` (one multiplication per entry) | 1.56 / 1.90 / 2.05 | 1.02 / 1.81 / 5.20 | 1.09 / 3.46 / 5.77 |
   | master, NormL2 `prox!` | 1.39 / 1.62 / 1.63 | 1.02 / 1.65 / 3.06 | 1.06 / 2.66 / 3.52 |
   | `y .= a .* x` | 1.57 / 1.92 / 2.27 | 1.02 / 1.82 / 5.68 | 1.09 / 3.46 / 6.23 |
   | `copyto!` | 2.69 / 1.03 / 1.36 | 2.87 / 4.45 / 1.52 | 1.36 / 5.73 / 1.21 |

   The probes show the size of the effect, not its mechanism. The dependence on the low 12 address bits
   suggests the store-to-load address aliasing of x86 cores, but that alone does not predict the dependence on
   the page backing, and no hardware counters were read. Code placement also interacts with it: in the band
   master's two compilations differ by up to 15% on Zen 3 at the same offset. From 1024 bytes up the
   threshold, ridge and broadcast loops are within a few percent of their time at offset 0. **The penalty is not new to the
   package:** master's ridge and NormL2 `prox!` and a plain broadcast have it on every CPU, Zen 3 included.
   What is specific to Zen 3 is that master's NormL1 and ElasticNet loops, the slowest of these, lose the
   least there.
3. **Julia's allocator often puts equal-size arrays in the band; it is not a rule.** Two arrays of the same
   size allocated one after the other are frequently `bytes + 64` apart, so 64 bytes apart modulo 4096 when
   the byte length is a multiple of 4096. It depends on the allocation history of the process: on the
   workstation (`probes/alloc_offsets.jl`) power-of-two sizes from 256 KiB to 8 MiB mostly gave +64 after a
   large array had been freed and offset 0 in a fresh process, and the reviewer's runs of the same script gave other
   offsets for some of those sizes. In the probes on the cluster, the million-entry array pairs that Julia
   allocated itself were 64 bytes apart in 31 to 39 of 48 pairs per element type and Julia version on Zen 3,
   and 128 to 256 bytes apart in most of the rest. A length that is not a multiple of 4096 bytes does not avoid
   the band (the reviewer found +128 for 2^20 + 16 Float32 entries). An in-place call (`y === x`) is at
   offset 0. The formal campaigns did not record the addresses of their arrays.
4. **Inside the band A3 is slower than master on Zen 3 for arrays of 16K entries and more.** The median over
   the processes of A3/master at each of the ten offsets from 16 to 256 bytes (`probe5`, node a04-18), its
   largest value, and the number of offsets where it is above 1.05:

   | Julia | type | entries | NormL1: largest A3/master (at offset), offsets above 1.05 | ElasticNet: largest (at offset), offsets above 1.05 |
   |:--|:--|--:|:--|:--|
   | 1.13.4 | Float32 | 2^20 | 1.25 (+96), 6 of 10 | 1.39 (+128), 7 of 10 |
   | 1.13.4 | Float32 | 16,384 | 1.04 (+96), 0 of 10 | 1.08 (+128), 4 of 10 |
   | 1.13.4 | Float32 | 1024 | 0.72 (+80), 0 of 10 | 0.80 (+16), 0 of 10 |
   | 1.13.4 | Float64 | 2^20 | 1.26 (+96), 5 of 10 | 1.38 (+128), 8 of 10 |
   | 1.13.4 | Float64 | 16,384 | 1.29 (+96), 6 of 10 | 1.36 (+128), 7 of 10 |
   | 1.13.4 | Float64 | 1024 | 0.76 (+80), 0 of 10 | 0.78 (+80), 0 of 10 |
   | 1.10.13 | Float32 | 2^20 | 1.35 (+64), 2 of 10 | 1.29 (+64), 2 of 10 |
   | 1.10.13 | Float32 | 16,384 | 1.37 (+64), 3 of 10 | 1.25 (+64), 2 of 10 |
   | 1.10.13 | Float32 | 1024 | 0.72 (+16), 0 of 10 | 0.65 (+80), 0 of 10 |
   | 1.10.13 | Float64 | 2^20 | 1.07 (+128), 1 of 10 | 1.03 (+128), 0 of 10 |
   | 1.10.13 | Float64 | 16,384 | 1.07 (+128), 1 of 10 | 1.03 (+128), 0 of 10 |
   | 1.10.13 | Float64 | 1024 | 0.72 (+16), 0 of 10 | 0.56 (+80), 0 of 10 |

   With Julia 1.13.4 A3 is slower than master at most offsets of the band at 2^20 entries, by up to 1.25×
   (NormL1) and 1.4× (ElasticNet); with Julia 1.10 at two or three offsets for Float32, by up to 1.37×. At 1024
   entries A3 is never slower. On the Intel nodes master pays the penalty as well and A3 stays equal or faster
   at nearly every offset: nothing above 1.05 on Broadwell with Julia 1.13.4, 1.11–1.25× at one or two offsets
   with Julia 1.10, and 1.06–1.07× at one to three offsets on Skylake-SP at 16,384 entries
   (`notes/tables/probe-band.md`). Single processes go further: Broadwell with Julia 1.10, ElasticNet at 1024
   Float32 entries and +16 bytes, is 1.21–1.27× in three of four processes and 0.51× in the fourth.
5. **At +64 bytes, the allocator's offset**, A3/master, and at offset 0 for comparison (range over the
   processes of three Zen 3 nodes, two Broadwell nodes and one Skylake-SP node; the number of processes of each
   cell is in `notes/tables/probe-offset64.md`; a cell is a ratio of the two medians over the rounds):

   | CPU, Julia | type | entries | NormL1 at +64 | at 0 | ElasticNet at +64 | at 0 |
   |:--|:--|--:|:--|:--|:--|:--|
   | Zen 3, 1.13.4 | Float32 | 2^20 | 0.96–1.25 | 0.83–0.84 | 1.17–1.37 | 0.83–0.84 |
   | Zen 3, 1.13.4 | Float64 | 2^20 | 0.91–1.08 | 0.83–0.84 | 1.18–1.63 | 0.83–0.84 |
   | Zen 3, 1.13.4 | Float32 | 16,384 | 0.68–1.24 | 0.67–0.69 | 0.74–1.29 | 0.72–0.73 |
   | Zen 3, 1.13.4 | Float64 | 16,384 | 0.96–1.24 | 0.67–0.68 | 0.97–1.64 | 0.70–0.72 |
   | Zen 3, 1.13.4 | both | 1024 | 0.67–0.68 | 0.67 | 0.69–0.72 | 0.69–0.70 |
   | Zen 3, 1.10.13 | Float32 | 2^20 | 1.35–1.38 | 0.73–0.74 | 1.27–1.38 | 0.65–0.67 |
   | Zen 3, 1.10.13 | Float64 | 2^20 | 0.83–0.86 | 0.74–0.75 | 0.86–0.92 | 0.65–0.68 |
   | Zen 3, 1.10.13 | Float32 | 16,384 | 0.60–1.38 | 0.61–0.63 | 0.58–1.28 | 0.58–0.59 |
   | Zen 3, 1.10.13 | Float64 | 16,384 | 0.59–0.85 | 0.60–0.61 | 0.54–0.87 | 0.54–0.57 |
   | Zen 3, 1.10.13 | both | 1024 | 0.51–0.54 | 0.58–0.59 | 0.48–0.53 | 0.53–0.55 |
   | Broadwell, 1.13.4 | both | 1024 to 2^20 | 0.65–0.92 | 0.65–0.98 | 0.67–0.93 | 0.67–0.98 |
   | Broadwell, 1.10.13 | both | 1024 to 2^20 | 0.44–0.85 | 0.61–0.91 | 0.48–0.97 | 0.60–0.88 |
   | Skylake-SP, 1.13.4 | both | 2^20 | 0.93–0.99 | 1.00–1.03 | 0.96–1.01 | 0.99–1.01 |
   | Skylake-SP, 1.13.4 | both | 1024 and 16,384 | 0.77–1.05 | 0.76–0.78 | 0.81–1.06 | 0.81–0.87 |
   | Skylake-SP, 1.10.13 | both | 1024 to 2^20 | 0.48–0.86 | 0.45–1.01 | 0.54–0.89 | 0.54–0.98 |

   On the arrays that Julia allocated itself in the processes of `probe1` (2^20 entries, nearly all in the
   band): NormL1 0.84–1.08 and ElasticNet 0.82–1.12 in Float32, NormL1 0.87–1.07 and ElasticNet **1.01–1.46** in
   Float64, with Julia 1.13.4; NormL1 0.73–1.25 and ElasticNet 0.65–1.23 in Float32 and at most 1.00 in Float64
   with Julia 1.10. The check-free kernels with A3's formula and with `clamp` lose as much in the band on Zen 3
   as A3 does, and the kernel with master's formula as little as master.
6. **This is the likely explanation of §8.6, not a proven one.** The harness allocates the input and the output
   of a case one after the other (`y = similar(x)`), and its sizes are powers of two, so its million-entry
   cases can be in the band in some processes and out of it in others; the lower level that A3 took there,
   91 µs, is that of the probes outside the band, and the higher ones are in the range of the probes inside it. The campaigns did not record
   pointers, so no single process of `formal6` or of `formal4` (91 against 109 µs in all six processes on two
   nodes) is attributed by measurement. Below 2^20 entries the cases of the harness were stable in every
   campaign.
7. **Consequence for A3.** Outside the band, and on 4 KiB pages, A3 is 0.65–0.87 of master on Zen 3. In the
   band on huge pages, where the allocator often places two consecutively allocated large arrays of a
   power-of-two size on these nodes, A3 on Zen 3 ranges from a smaller gain to a loss of up to 1.3× (NormL1)
   and 1.6× (ElasticNet) in single processes with Julia 1.13.4, and up to 1.4× for Float32 with Julia 1.10.
   How often a caller is there depends on the allocator, the sizes and the huge-page setting of the machine;
   the probes do not give that frequency (a uniform offset would be in the band 6% of the time).
8. **B925 stays ahead of master in the band** (`probe7`, the same sweep with a step array in a third region at
   the offset of the input, so the output moves against both inputs together and not against each
   separately; two processes per stratum). Its vectorized loop loses up to 1.3–1.6× against itself in the band on Zen 3 and
   up to 4× on Broadwell with Julia 1.10, but it starts 2.5 to 50 times ahead of master's scalar loop: B925/master
   is at most 0.43 at every offset on Zen 3 (0.16–0.36 at offset 0 with Julia 1.13.4), at most 0.55 on
   Broadwell and at most 0.80 on Skylake-SP. One cell is above 1: Broadwell, Julia 1.10, Float64, 1024 entries,
   1.11× at +48 bytes and above 1.05 at two of ten offsets, against 0.28 at offset 0
   (`notes/tables/probe-band-b925.md`). V has no output array, and D3 removes a scan and adds no loop, so the
   finding does not apply to them.

Limits, from the fourth review and my own: offsets were swept in ascending order and the trees compiled in one
order; a row is the median or minimum over the rounds, and the rounds were not saved; the A/A acceptance rule of
§2.2 was not applied to the probes; `layout_probe.jl` looks up `prox!` through a module on every call (32 bytes
and a dynamic dispatch, which matters at 1024 entries and not at 100 µs); the probes do not check the outputs,
which the harness of §2.2 does for the same trees; and the reviewer's machine is not a Zen 3, so the Zen 3
ratios were recomputed from the CSVs, not reproduced.

### 8.8 Repeated inputs: master's scalar soft threshold on Julia 1.10 has data-dependent branches

On Julia 1.10 the scalar code of `shrink` on master tests the sign bits of `v - c` and of `v + c` with two
conditional jumps (`probes/scalar_branches.jl`; Julia 1.13 compiles the same source without them; the reviewer
confirmed both, also in the full sparse ElasticNet loop). The ordinary path of A3 is `vmaxsd`, `vminsd`, a
subtraction and an addition on both versions; its range tests and its exceptional path keep their branches. A
benchmark that calls a scalar loop again and again on one input lets the predictor learn the two jumps.

`bench/sparse_probe.jl` runs `prox!` on a sparse iterate (65,536 entries, 6,553 stored; the case
`enet/sparse_prox!/scalar` of `ridge910.jl`) and replaces the stored values before every call by one of K value
sets of independent normal values. Time of one call on master in µs, and A3/master, Julia 1.10.13, Float32
ElasticNet, two processes. The times are a call with its copy of the values minus a separately timed copy
(0.4 to 0.5 µs), so they are approximate; the raw times show the same thing (master 11.8 µs at K = 1 and
50.2 µs at K = 512, A3 14.1 and 14.8 µs):

| | K = 1 | K = 2 | K = 8 | K = 64 | K = 512 |
|:--|:--|:--|:--|:--|:--|
| Zen 3: master | 10.5–11.3 | 10.5–11.0 | 41–42 | 49 | 50 |
| Zen 3: A3/master | **1.21–1.34** | 1.24–1.34 | 0.33–0.34 | 0.27–0.28 | 0.28–0.29 |
| Broadwell: master | 43 | 50–51 | 57 | 57 | 57 |
| Broadwell: A3/master | 0.41 | 0.36 | 0.31–0.32 | 0.31–0.32 | 0.31 |

- NormL1, ElasticNet with weights and Float64 behave the same way: on Zen 3 A3/master is 1.04–1.34 at K ≤ 2
  and 0.24–0.35 from K = 8; on Broadwell it is 0.31–0.50 for every K.
- The table is from `probe3` (node b09-17), where every tree had its own arrays. `probe6` repeated it on
  another node (a04-18, three processes) with the arrays shared between the trees: 1.03–1.35 at K ≤ 2 and
  0.23–0.36 from K = 8, master at 10.6–16.1 µs and 40.6–66.6 µs.
- With one or two value sets the Zen 3 predictor has learned master's branches (10.5 µs); from eight sets on
  master takes 41–50 µs and A3, which takes about 14 µs for any K, is 3 to 3.6 times faster. Broadwell learns
  little of the sequence (43 µs at K = 1 against 57 µs from K = 8).
- The reviewer excluded the cache footprint of the value sets as the cause on his machine: with 512 value sets
  that hold the same values master takes 7.7 µs, with 512 different ones 35.5 µs, and A3 7.2 µs in both. That
  control was not run on Zen 3, and no branch-miss counters were read there.
- **The 1.15–1.22× of `ridge910` on Zen 3 with Julia 1.10 (§8.6) is the K = 1 column**: a repeated input on a
  CPU that can memorize 6,553 outcomes. Values that change like these random sets give the other columns.
  Neither is the wrong workload: a caller can repeat an input, and changing values can keep their signs, and
  with them the branch outcomes. The K = 1 reading is a real 1.2–1.35× for that caller.
- The branches are in every scalar soft-threshold loop of master on Julia 1.10 (`prox_block` loops, step
  arrays, sparse and symmetric iterates); only the sparse loop was probed. They would explain why A3's Julia
  1.10 ratios on step arrays fall with the
  length in §8.6 (0.70–0.77 at 16 to 1024 entries, 0.12 at 2^20 on Zen 3): the longer the array, the less of it the
  predictor holds.
- **With Julia 1.13.4 neither tree depends on K**, and the comparison on Zen 3 is not resolved. With the input
  and the output arrays shared between the trees (`probe6`, three processes, node a04-18), A3/master is
  0.94–1.18 for ElasticNet, 0.95–1.02 for NormL1 and 0.89–1.09 with weights, while master's second compilation
  reads 0.86–1.13 against its first on the same arrays. The loop takes 3 to 4 ns per stored entry there on
  master, nearly twice what A3 needs on Julia 1.10 on the same CPU, and moves by 10% with code placement.
  On Broadwell A3 is 0.92–0.97 in every process, with the control at 0.98–1.01.

## 9. Recommendations

Ranked. Nothing here was merged; every patch is a branch with tests (§7), and rows 1 to 4 are the draft pull
requests #966 (D3), #967 (V), #968 (B925) and #969 (A3). The 5% rule is the
maintainer's: a reproducible slowdown above 5% on an existing ordinary path needs an explicit ruling.

| # | Item | Recommendation | Gain | What the maintainer has to accept |
|--:|:--|:--|:--|:--|
| 1 | **D3**: skip the second scan of a zero group when the threshold decides (1 file, +15 −1, bit-identical) | **Implement** | 0.83–0.95 on NormL2 and group cases with zero or sparse groups, every CPU and both Julia versions; 18 to 48 cases faster in every process per stratum; none slower at 16 entries and above | Zero groups of 1 to 5 entries at a threshold below the guard (level `floatmin`): 1.10–1.15× on Broadwell with Julia 1.13.4, about 0.3 ns per group. One 3-entry Float32 NormL2 `prox!` cell on Broadwell with Julia 1.10 (1.06–1.08×). Both are above 5%, so both need the ruling. |
| 2 | **V**: one loop for unweighted NormL1 and NormL2 values of 16 to 1024 entries (2 files, +27 −3) | **Implement if the last-bit change is acceptable** | Values 0.62–0.87 (geometric means); 11 to 30 cases faster per stratum | **A numerical ruling:** inside the window the value can differ from master's in the last place (at most 1 eps in 6 of about 5,900 harness probes), with the same error growth. Outside it the bits are master's. NormL2 value of 1 or 2 entries: 1.06–1.13× (Broadwell; 1.10× for 2 Float32 entries on Zen 3 with Julia 1.10), a fraction of a nanosecond. |
| 3 | **B925**: #925's design for NormL1 step arrays (1 file, +34, bit-identical) | **Investigate further: one revision, then propose** | Step arrays 0.10–0.45 at 16 entries and above (geometric means); 0.03–0.15 at 65,536 entries on Julia 1.10 | The costs #925 was given for ElasticNet: 1 to 9 entries 1.06–2.55× (a few nanoseconds), and the generic loop 1.01–1.13× when a step is exceptional. The third review asks to skip the pass for short arrays first; that revision is not written. In the placement band of §8.7 it stays ahead of master (one cell at 1.11×); the sweep moved the output against the input and the step array together, not against each. |
| 4 | **A3**: select-based soft threshold on the ordinary NormL1 path (2 files, +20 −1, same values as `shrink` for every non-negative threshold) | **Do not merge under the current policy.** Ready if the maintainer rules otherwise. | Dense `prox!` and `value_and_prox!` of NormL1 and ElasticNet with a scalar step: geometric means 0.65–0.97 at 16 entries and above on every CPU (§8.6). Julia 1.10: every scalar path loses its branches, 3 to 3.6× faster on inputs that change between calls. | **Reproducible costs.** (a) Zen 3, huge pages, output 16 to 256 bytes after the input modulo 4096, where the allocator often places large power-of-two arrays: up to 1.3× (NormL1) and 1.6× (ElasticNet) slower at 16K entries and more; all six processes of the formal run on Julia 1.10 at 2^20 Float32 entries, 1.20–1.29× (§8.6, §8.7). (b) Julia 1.10, lengths 1 to 24: 1.05–1.38×, 1 to 2.5 ns (§8.2, §8.6). (c) Julia 1.13: `prox_block` loops of ElasticNet over 4, 12, 16 and 100 entries, 1.04–1.19×, and NormL1 `prox!` of 1 and 4 Float32 entries on Broadwell, 1.07–1.12× (§8.6). (d) A repeated sparse input on Zen 3 with Julia 1.10: 1.15–1.35× (§8.8). |
| 5 | **Draft PR #155** (not from this study; measured in §8.5) | **Investigate further before merging** | Weighted NormL1 value 4.5–9× faster on Julia 1.10, 0.70–0.80× on Julia 1.13 with AVX2 | 1.14–1.28× slower on the AVX-512 node with Julia 1.13.4 in all six processes (7 cases). Its own measurements were on Zen 3 only. |
| 6 | Least squares; NormL2 and group shrinkage of 1024 entries; the scalar-step structure of the ℓ1 loops; the range checks of small groups | **Keep** | | They are at the floor of their check-free kernels (§4), or the gap is the guarantees (§7.6). BLAS `nrm2`, Blue's algorithm and ProximalOperators are slower or less accurate where they differ (§4, §6). |
| 7 | A, B and B2, C and C2, D and D2, F, the fused `value_and_prox!`, a cached Cholesky for the least-squares prox, A3's formula at ElasticNet's own `shrink` calls | **Reject** | | §7.5 and §7.1: each has a measured regression or an accuracy loss that its gain does not pay for. |

**Why A3 moved from "propose" to "do not merge".** Until the confirmation on the new master, A3 had no slower
case at 16 entries and above in any stratum, and its known cost was a nanosecond or two at a few lengths on
Julia 1.10. The confirmation then gave a slower case in every process (ElasticNet at 2^20 Float32 entries on
Zen 3 with Julia 1.10), and the probes of §8.7 showed that it is not noise and not one node: A3's loop is fast
enough to expose a placement penalty that master's slower loop mostly hides on Zen 3, and the default allocator
produces that placement for the array sizes benchmarks like best. On the Intel nodes master pays the penalty
too, and A3 stays equal or faster at nearly every offset (§8.7).
In most placements measured A3 is a gain, or neutral, on every machine; under a rule about reproducible
slowdowns it is a patch with four kinds of them. That is a policy decision, so it goes to the maintainer with
the numbers. The fourth review reached the same recommendation.

- **None of the formulas tried avoids cost (a).** On Zen 3 the check-free kernels with A3's formula, with
  `clamp` and with select-based `max` and `min` lose as much in the band as A3, and the kernel with master's
  formula loses as little as master (§8.7). That is four formulas, not a proof that no fast form is robust. A
  call-time test of the two pointers that sends the band to the old loop would recover both, at the price of
  machine-specific dispatch in a proximal operator. I did not build it and do not recommend it.
- **If the maintainer takes A3**, the open design question of §7.1 comes with it: two soft-thresholding
  functions, or `shrink` itself with the new formula.
- **D3 and V do not depend on A3**, and neither does B925. D3 and V each have cells above 5% as well (rows 1 and 2); they are a fraction of a nanosecond at 1 to 5 entries, and they need the same explicit acceptance. FINAL (= A3 + V + D3) was measured only to show that
  the three do not interfere; it inherits A3's cases.
- **The gains do not reach solver iterations of regression workloads** (§8.3: 0.99–1.00). They matter where the
  proximal step is the iteration.

**For the benchmarks rather than the library** (§10 items 1, 2, 6 and 7): capture function values instead of
modules in the probes, fix `prox_block1`, do not time arrays that are 64 bytes apart modulo 4096 without
knowing it, and cycle scalar-loop probes through several inputs on Julia 1.10. The last two change the sign of
some tree-to-tree comparisons, including one of this study's.

**Julia 1.10 regressions above 5% in the proposed set (D3, V).** None at 16 entries and above. Below that: V's
Float32 NormL2 value of 1 or 2 entries (1.08–1.13×, a fraction of a nanosecond) and D3's 3-entry cell above. The
benefit on Julia 1.13.4 is 0.74–0.87 on V's cases and 0.83–0.95 on D3's (geometric means). A3's Julia 1.10
regressions are its costs (a) and (b); its Julia 1.13.4 benefit is 0.73–0.96 on its cases outside the band.

## 10. Other findings for the maintainer

None of these is changed by a patch of this study.

1. **Probes of `benchmark/l2_norms.jl` and of the NormL1 harnesses dispatch dynamically.** They call the package
   through a captured module (`() -> PB.prox_block!(...)`), which is a lookup and a dynamic dispatch on every
   call: about 23 ns and 64 bytes on the workstation, where the call itself takes 5.5 ns. The reviewer
   confirmed it (31–60 ns and 64 bytes against 5–11 ns and none). Single-call and per-group probes are dominated
   by it, so tree-to-tree ratios of small kernels are compressed toward 1 and the probes report allocations
   that the package does not make. Capturing the function value instead of the module removes it.
2. **`prox_block1` in `benchmark/l2_norms.jl` times another input than the one it checks.** The probe captures
   variables that `block_update_sweep` reassigns (found by the reviewer, with a two-group reproduction).
3. **Julia 1.10: the weighted NormL1 value loop is not vectorized on master** (1.8–7.4× slower than on Julia
   1.13 in Float64, 2.1–14× in Float32; §5). Draft PR #155 fixes it, with the AVX-512 cost of §8.5.
4. **Julia 1.10: a second failure in `test/numerical_kernels.jl`** (lines 1018–1019, "group value below
   K floatmin with zero-weight groups: Float64") is hidden behind the known one at line 735, because a failing
   top-level testset stops the file. Both are on master `18b27149`.
5. **`value_and_prox!` of an unweighted NormL1 with a Float64 level and Float32 data can differ from `f(y)` in
   the last place.** The two use different value loops.
6. **Benchmarks that allocate the input and the output one after the other, with power-of-two sizes, often
   measure the placement band** (§8.7). The two arrays are then frequently 64 bytes apart modulo 4096, where,
   on huge pages, master's NormL1 loop is up to 1.3–1.5× slower on Zen 3, 1.5–1.8× on Broadwell and
   1.4–3.6× on Skylake-SP than at any offset outside 16 to 256 bytes, and its ridge loop up to 2× and 5.8×. `ridge910.jl` uses 1024 and 65,536
   entries and, for its sparse probes, one set of arrays per role; a role whose arrays land in the band and one
   whose arrays do not differ by that much with identical code. That is a candidate explanation, which I did
   not verify, for the role-dependent A/A controls of §8.4 and §8.6. Changing the length does not reliably avoid
   it; recording `pointer(y) - pointer(x)` in a harness, or placing the arrays deliberately as
   `bench/offset_probe.jl` does, shows where a case is.
7. **Benchmarks that repeat one input measure a learned branch sequence on Julia 1.10** (§8.8). On Zen 3 the
   scalar soft-threshold loops of master are 10.5 µs for 6,553 stored entries on a repeated input and 41–50 µs
   when the values change between calls. A tree-to-tree ratio from such a probe can have the wrong sign: A3
   reads 1.04–1.34× on the repeated input and 0.24–0.35× on random value sets. Cycling through eight or more
   value sets removed the effect in this probe.
8. **ElasticNet with a step array is still 3 to 4 times a bare loop after #925** (`screen3`, Broadwell: a
   check-free kernel is 0.35× of master on Julia 1.13 and 0.25× on Julia 1.10, at 16 to 2^20 entries). The rest
   is the ridge weight per entry: a division and its range tests. This is reported for completeness; the
   maintainer has closed that line of work with #925.

## 11. Reproduction

```sh
# check-only run of every implementation and tree (local)
KO_TREES="base=/path/to/master;FINAL=/path/to/tree" KO_OUTPUT=out KO_CHECK_ONLY=1 \
  julia --startup-file=no --threads=1 --check-bounds=no --project=env bench/run.jl
# a formal comparison on one Discovery node (KO_ROOT holds trees/<name>, experiment_kernelopt/ and the Julia environments)
sbatch --partition=main --constraint=epyc-7513 --exclusive --mem=0 \
  --export=ALL,KO_ROOT=$HOME/projects/pbopt-m3,KO_TAG=formal6,KO_KIND=epyc7513,KO_REF=base,KO_TREE_LIST="base A3 V D3 B925 FINAL",KO_TREES_ONLY=1,KO_SUITES=l1,l2,group,enet,solver,KO_PROCS=6,KO_JULIAS=1.13.4 \
  slurm/formal.sbatch
julia bench/digest.jl results/m3/formal6                 # the tables of sections 8.1 and 8.6
python3 bench/slowcells.py results/m3/tiny3 --list       # tiny inputs, with the cells a tree cannot affect
python3 bench/ridge910_digest.py results/m3/native5 epyc7513-julia1.10.13-FINAL
# placement and repeated-input probes (section 8.7, 8.8)
sbatch --exclusive --mem=0 --export=ALL,KO_ROOT=...,KO_TAG=probe5,KO_KIND=epyc7513,KO_REF=base,KO_CAND=A3,KO_SCRIPT=offset_probe.jl slurm/probe.sbatch
python3 bench/offset64.py results/m3 --report; python3 bench/offset_summary.py results/m3/probe2 results/m3/probe5
python3 bench/sparse_digest.py results/m3/probe6
julia bench/numerics.jl                                  # section 6 (KO_TREES as above)
```

Files: `bench/` (harness, suites, solver loops, probes, numerics tables, digests), `kernels/` (standalone
kernels), `slurm/` (job scripts), `probes/` (the exhaustive, code-generation and allocator checks cited above),
`notes/` (inventory, prior findings, reviews, tables), `results/` (not in this reading copy: raw CSV, provenance, Slurm outputs; the
outputs of one faulty probe version are kept apart in `results/m3/probe-invalid/` with a note and are not used).

Published branches: `kernelopt/d3-zero-group-scan` (#966), `kernelopt/v-value-loop` (#967),
`kernelopt/b925-l1-step-arrays` (#968), `kernelopt/a3-select-soft-threshold` (#969), `kernelopt/report` (this copy).

Measured trees (local worktrees under `pbopt/`, branches `exp/kernelopt-*`, not pushed): `base2` = master `18b27149`;
`A3r` `26d61415`; `Vr` `453906c8`; `D3r` `1a783f47`; `B925` `741cd0ac`; `FINALr` `7da6f7df`.
