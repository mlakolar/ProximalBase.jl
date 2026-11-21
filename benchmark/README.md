# Minimal-core benchmark and independent references

The 120-case harness compares the retained storage/function hot paths to the
full post-#848 source at eff21b6d7d33c622313122f46c6ee71693c101ee. It includes
an independently compiled same-source A/A control, alternating roles/order,
Float32/64, state restoration outside timing, warmup, raw medians/minima and
allocation counts. Historical def81ec9 is a separate diagnostic reference;
this reduction makes no new primary performance claim against that revision.

From the package root:

```sh
mkdir -p /tmp/pb849-baseline
git archive eff21b6d7d33c622313122f46c6ee71693c101ee src | tar -x -C /tmp/pb849-baseline
julia scripts/instantiate.jl benchmark
PB_BASELINE=/tmp/pb849-baseline PB_CANDIDATE="$PWD" PB_CHECK_ONLY=1 julia --project=benchmark benchmark/run.jl
PB_BASELINE=/tmp/pb849-baseline PB_CANDIDATE="$PWD" PB_OUTPUT=/tmp/pb849-results julia --threads=1 --check-bounds=no --project=benchmark benchmark/run.jl
python3 benchmark/summarize.py /tmp/pb849-results
```

Set PB_ROUNDS=21 for follow-up measurements and PB_FILTER to a case-name regular
expression for focused investigations. PB_CHECK_ONLY runs the cross-implementation
probe checks without timing; CI executes it against the frozen tree. The separate
differential.jl evaluates independent dense/BigFloat mathematical references;
its current-package checks are also part of test/minimal_math.jl.

Run timing without other Julia/CPU-heavy work. Judge reproducible slowdowns
above five percent using the paired distribution, A/A controls, allocations,
fresh-process repeats and reversed source roles. Evidence and the performance
review for #849 are recorded under docs/audit/minimal-core-849.

The archived measurement environment contains its original Project/Manifest as
provenance, including an unused local ProximalBase path inherited from the
verification environment. Drivers load both source trees explicitly through
the PB_BASELINE/PB_CANDIDATE variables. For an exact dependency replay, update
only that unused path to a restored checkout before instantiating the archived
environment; preserve the pinned BenchmarkTools/PrecompileTools versions.
The active benchmark/Project.toml is a clean environment without that path.

## Specializing timed probes

Resolve package callables from each role's module while building the probe. Pass
the callable and its data to a helper that returns the timed closure with concrete
captures. For a sweep, pass the callable and the sweep data to the same helper.

Warm the operation before measuring it in a function such as
`warmed_bytes(op::F) where {F} = (op(); op(); @allocated op())`.
For probes with a setup step, restore the input before each call and measure
only the operation. Run a filtered `PB_ROUNDS=1` smoke test and inspect the
`memory` and `allocs` columns of `raw.csv` for X, Y and XC.

Archived results remain valid records of their runs. Dispatch and allocations in
the original harnesses can dilute paired ratios toward 1 for fast cells.

## NormL1 hot-loop harness

`norml1_infinite_weight.jl` times the `NormL1` proximal loops (weighted and
unweighted, scalar and per-entry steps, dense, sparse and block storage, Float32 and
Float64) between two source trees, with an independently compiled A/A control, and
`summarize_norml1_infinite_weight.py` reports the per-round ratios. It was written for
issue #853 and is meant for #866 and any numerical change that touches these loops.
Run it with the roles reversed and in fresh processes, and judge slowdowns above five
percent against the A/A column. The summarizer pools only runs with the same X and Y trees;
summarize a reversed-role run on its own (its ratios are the inverse). The usage is in the header of the script;
`PB_CHECK_ONLY=1` checks that the outputs agree without timing.

```sh
PB_FIRST=/path/to/tree-before PB_SECOND="$PWD" PB_CHECK_ONLY=1 \
  julia --threads=1 --project=benchmark benchmark/norml1_infinite_weight.jl
```

## NormL1 value harness

`norml1_value_underflow.jl` times the `NormL1` value paths (the dense weighted SIMD loop, mixed zero
weights, small ordinary values, the zero vector, unweighted values, views, sparse, block and symmetric
iterates and `value_and_prox!`) and the recomputation of sums whose terms round below `floatmin`, for
Float32 and Float64 at n = 100, 1024, 65536 and 100000. It was written for issue #373 and has the design
of the hot-loop harness above (roles X, Y and the A/A control XC, alternating order, reversed-role and
fresh-process repeats), and `summarize_norml1_infinite_weight.py` summarizes its output. The `underflow`
probes are the recomputed sums, whose values the fix changes; the other outputs must agree.

```sh
PB_FIRST=/path/to/tree-before PB_SECOND="$PWD" PB_CHECK_ONLY=1 \
  julia --threads=1 --project=benchmark benchmark/norml1_value_underflow.jl
```

## ElasticNet harness

`elasticnet.jl` times `ElasticNet` (issue #908) between the archived source tree at
eff21b6d7d33c622313122f46c6ee71693c101ee (`PB_ARCHIVE`), the candidate (`PB_CANDIDATE`) and a
second compilation of the candidate as the A/A control: a loop of `prox_block`, dense `prox!`
(unweighted and weighted, a zero `λ1` or `λ2`, per-entry steps) at n = 1024 and 65536, dense
values (unweighted, weighted, and the two regimes where one part dominates), `value_and_prox!`,
`SparseIterate`, `BlockIterate` and `SymmetricSparseIterate` values and `prox!`, and the `prox!`
of a `Scaled` `ElasticNet`, in Float32 and Float64. The roles share their plain input and output
arrays. Each round times every case in alternating role order and keeps the minimum over the
samples of each role; the summary gives per case the median, minimum and maximum of the paired
ratios candidate/archive and control/candidate, with the warmed allocations, and marks medians
above 1.05. `PB_REVERSE=1` loads and times the candidate in the place of the archive, for the
reversed-role rerun in a fresh process; `PB_FILTER` selects cases; `PB_CHECK_ONLY=1` compares
the outputs only (proximal points within 4 ulp, values within `2(n + 2) eps`).
Measured with Julia 1.13.3 and 21 rounds, the four `prox!/array_step_weighted` cases (Float32 and
Float64, n = 1024 and 65536) exceed the five percent threshold: median ratios 1.76 to 1.86 on a shared
machine and 1.93 to 2.00 on a machine in exclusive use (AMD EPYC 7542), each reproduced by a reversed-role
run; the per-entry ridge factor of an array of steps is tracked in #910.

```sh
PB_ARCHIVE=/tmp/pb849-baseline PB_CANDIDATE="$PWD" PB_ROUNDS=21 PB_OUTPUT=/tmp/pb908-results \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/elasticnet.jl
```

## NormL1Nonneg harness

`norml1nonneg.jl` times `NormL1Nonneg` (issue #947) between the archived source tree at
eff21b6d7d33c622313122f46c6ee71693c101ee (`PB_ARCHIVE`), the candidate (`PB_CANDIDATE`) and a
second compilation of the candidate as the A/A control, in Float32 and Float64 at n = 8, 1024,
2048 and 65536: a loop of `prox_block`, dense `prox!` (unweighted and
weighted, a zero level, per-entry steps), dense values at a feasible point (unweighted, weighted,
a zero level), `value_and_prox!`, `SparseIterate`, `BlockIterate` and `SymmetricSparseIterate`
values and `prox!`, and the `prox!` and `value_and_prox!` of a `Scaled` `NormL1Nonneg`. A dense call
takes its argument in turn from four arrays, so that consecutive calls do not repeat their input;
the roles share their plain input and output arrays. Cases whose id starts with `info/` time the value at an
input with a negative or a `NaN` entry, in every round; their outputs are not compared with the archive, the A/A
check applies to them, and they are not part of the comparison of the plan. Each round times every case in
alternating role order and keeps the minimum over the samples of each role; the summary gives per case the median,
minimum and maximum of the paired ratios candidate/archive and control/candidate, with the warmed
allocations, and marks medians above 1.05. `PB_REVERSE=1` loads and times the candidate in the place
of the archive, for the reversed-role rerun in a fresh process; `PB_FILTER` selects cases;
`PB_CHECK_ONLY=1` performs the output checks and exits before timing; the control must be identical to the
candidate. With `PB_VS_NORML1=1` the archive role is `NormL1` of the
candidate tree.

```sh
# the archived source tree at eff21b6d, extracted once
mkdir -p /tmp/pb947-archive && git archive eff21b6d7d33c622313122f46c6ee71693c101ee src | tar -x -C /tmp/pb947-archive
PB_ARCHIVE=/tmp/pb947-archive PB_CANDIDATE="$PWD" PB_ROUNDS=21 PB_OUTPUT=/tmp/pb947-results \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/norml1nonneg.jl
```

## IndNonnegative harness

`indnonnegative.jl` times `IndNonnegative` (issue #948) between the archived source tree at
eff21b6d7d33c622313122f46c6ee71693c101ee (`PB_ARCHIVE`), the candidate (`PB_CANDIDATE`) and a
second compilation of the candidate as the A/A control, in Float32 and Float64: a loop of
`prox_block`, dense `prox!` with a scalar step and with an array of steps, dense values at a
feasible input and at an input with one negative entry (the last), `value_and_prox!` (a scalar
step and an array of steps), `SparseIterate`, `BlockIterate` and `SymmetricSparseIterate` values
and `prox!`, at n = 1024 and 65536. The roles share
their plain input and output arrays, and the summary has the columns of the ElasticNet harness
(median, minimum and maximum of the paired ratios candidate/archive and control/candidate, warmed
allocations of the archive and candidate roles, with those of the control in `raw.csv`, medians above 1.05
marked). A case id starts with the element type `<T>`, Float32 or Float64. `PB_REVERSE=1`, `PB_FILTER`, `PB_ROUNDS` and `PB_OUTPUT`
act as there.

The archive does not evaluate the value at the output of `value_and_prox!` (its fused value is
always 0), so the cells `<T>/value_and_prox!/...` report the ratio to the archive without a verdict, and
the cells `<T>/comparator/value_and_prox!/...` are the ones judged. In a comparator cell the role
"archive" is the candidate's own `prox!` followed by `f(y)` and the roles "candidate" and "control" are
`value_and_prox!` of their trees, so that the C/A column is the ratio of the candidate's
`value_and_prox!` to its `prox!` then `f(y)`, and the A/A column is as in the other cells.

The inputs have no `NaN` and their negative entries lie far below the tolerance of the archived
value (`sqrt(eps(T))`). `PB_CHECK_ONLY=1` compares the outputs bit for bit: proximal points, values,
the fused values, the comparator cells against `value_and_prox!`, and the control against the
candidate. The three intended differences from the archive (the tolerance of its value, its value
`Inf` at a `NaN` entry where the candidate's is `NaN`, and its fused value, always 0) are outside
the timed inputs.

```sh
PB_ARCHIVE=/tmp/pb849-baseline PB_CANDIDATE="$PWD" PB_ROUNDS=21 PB_OUTPUT=/tmp/pb948-results \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/indnonnegative.jl
```

## MCP harness

`mcp.jl` compares the candidate with the archived tree at
`eff21b6d7d33c622313122f46c6ee71693c101ee` and a second compilation of the
candidate for the A/A control. It uses Float32 and Float64 at sizes 8, 1024
and 65536. Steps range from `a/2` and `7a/8` to `9a/8` and `3a/2`.
Cases cover scalar `prox_block` loops, dense `prox!`, dense values, and
`value_and_prox!` with scalar and per-entry steps, plus value and scalar-step
calls on sparse, block and symmetric sparse iterates.

Seeded finite inputs stay more than 1.5% from decision boundaries in the
mathematically comparable cases. The harness checks each result against a
Float64 reference and checks fused values against the value at the stored
output. It measures warmed allocations before timing. `PB_CHECK_ONLY=1`
runs those checks alone and `PB_FILTER` selects case IDs by regular expression.

Each process alternates order over at least five paired rounds.
`PB_REVERSE=1` reverses the initial load and timing order in a fresh process.
`PB_OUTPUT` holds raw timings, the candidate/archive and A/A ratios for
each pair, allocation counts, provenance and a summary of medians and ranges.

```sh
mkdir -p /tmp/pb952-archive && git archive eff21b6d7d33c622313122f46c6ee71693c101ee src | tar -x -C /tmp/pb952-archive
PB_ARCHIVE=/tmp/pb952-archive PB_CANDIDATE="$PWD" PB_ROUNDS=7 PB_OUTPUT=/tmp/pb952-results \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/mcp.jl
PB_REVERSE=1 PB_ARCHIVE=/tmp/pb952-archive PB_CANDIDATE="$PWD" PB_ROUNDS=7 PB_OUTPUT=/tmp/pb952-results-reversed \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/mcp.jl
```

The extended sweep checks finite ranges and neighbors of the three decision
boundaries using the full-objective BigFloat reference from the MCP tests.

```sh
PB_MCP_SWEEP=large PB_MCP_SEED=952 julia --project=. --threads=1 benchmark/mcp_reference_sweep.jl
```

## ElasticNet array-step harness

`ridge910.jl` times the loop of `ElasticNet` with an array of steps (issue #910) and the code
around it, 76 cases in Float32 and Float64 at n = 1024 and 65536: `ElasticNet` with an array of
steps (dense `prox!` weighted, unweighted and with every 64th step outside the ridge band,
`value_and_prox!`, and a `SparseIterate`), `ElasticNet` with a scalar step (dense `prox!`, a
loop of `prox_block`, the `SparseIterate`), `SqrNormL2` loops of `prox_block` with and without
a center (array and scalar steps), scalar `prox!` and `value_and_prox!`, and neighbours that
#910 does not change (`SqrNormL2` value and `gradient!`, `NormL1` `prox!` with an array of
steps). The header of the script lists the cases.

The roles are a base tree (`PB_BASE`), one or more candidate trees (`PB_CANDIDATES`, a comma-separated
list of `name=root`) and a second compilation of the base as the A/A control. Pass a byte-identical
copy of the base as one more candidate: it measures the noise floor of a role that loads its own
copy of the same source. Each round times every role of every case in rotating order and keeps the
minimum over the samples of each role; the summary gives per case the median, minimum and maximum of
the paired ratios candidate/base and control/base, with the warmed allocations, and marks medians
above 1.05. Before timing, the script compares every candidate output with the base output, reports
the cases whose outputs are `isequal` (finite entries and signed zeros bit for bit, any two NaNs
equal) and stops at a difference above 4 ulp or at an entry that is not `isequal` and not finite in
at least one output.
`PB_REVERSE=1` loads the candidates before the base; `PB_FILTER` selects cases by a regular
expression on their ids; `PB_CHECK_ONLY=1` stops after the comparison.

```sh
BASE=$(mktemp -d); NOISE=$(mktemp -d); CAND=$(mktemp -d); OUT=$(mktemp -d)
git archive <base-sha> src | tar -x -C "$BASE"
git archive <base-sha> src | tar -x -C "$NOISE"
git archive <candidate-sha> src | tar -x -C "$CAND"
julia scripts/instantiate.jl benchmark
PB_BASE="$BASE" PB_CANDIDATES="C=$CAND,N=$NOISE" PB_CHECK_ONLY=1 \
  julia --threads=1 --project=benchmark benchmark/ridge910.jl
PB_BASE="$BASE" PB_CANDIDATES="C=$CAND,N=$NOISE" PB_ROUNDS=21 PB_OUTPUT="$OUT/forward" \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/ridge910.jl
PB_BASE="$BASE" PB_CANDIDATES="C=$CAND,N=$NOISE" PB_ROUNDS=21 PB_OUTPUT="$OUT/reversed" PB_REVERSE=1 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/ridge910.jl
```

Judge a slowdown above five percent only when it appears in both role orders and in fresh
processes, and compare it with the noise-floor candidate and the A/A control of the same process.

## ℓ₂ norms harness

`l2_norms.jl` times `GroupNormL2` and `NormL2` (issue #900) between two source trees that both define
them, `GroupNormL2` with the pure `prox_block!`. For `GroupNormL2`: the compact block proximal operator of one group (sizes 2, 5, 10, 32, 128 and 1024,
nonzero and zero trials), a sweep over every group, and the array `prox!`, `value_and_prox!` and value,
for Float32 and Float64, contiguous (range) and noncontiguous (permuted index vector) layouts, and dense,
partially zero and mostly zero arguments of about 4096 entries (`(4096 ÷ m)⋅m`), and a block-coordinate
sweep on a `BlockIterate` (for every group the trial from the iterate, the pure `prox_block!` into scratch
and the write `copyto!(blockview(x, k), out)`, after restoring the iterate by a whole-iterate copy). For `NormL2`: the
dense `prox!`, `value_and_prox!` and value at 1024 and 65536 entries, and the value of a 200 × 200
matrix `SparseIterate` and of a `SymmetricSparseIterate` from 400 random assignments (about 400 stored coordinates) on the ordinary path
and on the recovery of an overflowing sum of squares (#808), which the frozen tree computes over the
logical matrix. It has the design of the `NormL1` harnesses (roles X, Y and the
A/A control XC, alternating order, shared input and output arrays) and
`summarize_norml1_infinite_weight.py` summarizes it. The frozen full tree at
eff21b6d7d33c622313122f46c6ee71693c101ee (`legacy/full`) is the historical reference; its fused value is
rounded differently, so `PB_CHECK_ONLY=1` requires agreement to a few rounding errors between X and Y and
exact agreement between X and XC.

```sh
mkdir -p /tmp/pb-legacy && git archive eff21b6d7d33c622313122f46c6ee71693c101ee src | tar -x -C /tmp/pb-legacy
PB_FIRST=/tmp/pb-legacy PB_SECOND="$PWD" PB_CHECK_ONLY=1 \
  julia --threads=1 --project=benchmark benchmark/l2_norms.jl
```

### Bit-for-bit differential of `l2_norms.jl`

`PB_EXACT=1` with `PB_CHECK_ONLY=1` replaces the tolerance check of `l2_norms.jl` by an exact one: every
output of Y and of XC must equal that of X with `===` per array element (tuples element by element, scalars
by `===`), so signed zeros and NaN bit patterns count. It is meant for a refactor that must not change a bit
of any output; without it the check is the one described above.

```sh
git worktree add /tmp/pb-ref 769640859ca791d08a3f08a0faafadc137bb685a      # the reference tree, read only
PB_FIRST=/tmp/pb-ref PB_SECOND="$PWD" PB_EXACT=1 PB_CHECK_ONLY=1 \
  julia --threads=1 --project=benchmark benchmark/l2_norms.jl
```

## Group ℓ∞ norm harness

`groupnormlinf.jl` times `GroupNormLinf` (issue #956) between two source trees that both define it, through its exported methods:
the array `prox!` into a separate output, the in-place `prox!(x, f, x, γ)`, `value_and_prox!`, the value, and sweeps of the compact
`prox_block!` over all groups, with an output that is not the input and in place, for Float32 and Float64, groups of 4, 32, 512 and 4096
entries over 4096 entries (1024, 128, 8 and 1 groups), contiguous (range) and permuted index layouts. The two in-place cases destroy their
argument: the BenchmarkTools `setup` restores it (the argument, or every working copy before a sweep) outside the timing, and each timed
sample is one call (`evals = 1`) and up to 1600 samples are requested; the other cases request up to 100 samples of 16 calls.
Each is subject to the time limit of 0.15 s. The
weights give group `k` a radius that is 0.1, 0.4, 0.8 or 1.2 times its ℓ₁ norm (`k` modulo 4): with groups of 4, 32 and 512 entries three
groups in four are clipped and one is zero (fraction 1.2); with 4096 entries there is one group, with fraction 0.1, which is clipped, and
no group is zero. It has the design of `l2_norms.jl` (roles X, Y and
the A/A control XC, alternating order, shared input and output arrays, `raw.csv` and `provenance.txt` summarized by
`summarize_norml1_infinite_weight.py`). The frozen full tree at eff21b6d7d33c622313122f46c6ee71693c101ee (`legacy/full`) is the
historical reference for X. `PB_CHECK_ONLY=1` requires agreement between X and Y to `8 (m + K) eps(T)` times the largest entry of an
array (or relative to a value) for groups of `m` entries and `K` groups, and `isequal` outputs of X and XC; with `PB_EXACT=1` it
requires equality of X and Y bit for bit.

```sh
mkdir -p /tmp/pb-legacy && git archive eff21b6d7d33c622313122f46c6ee71693c101ee src | tar -x -C /tmp/pb-legacy
PB_FIRST=/tmp/pb-legacy PB_SECOND="$PWD" PB_CHECK_ONLY=1 \
  julia --threads=1 --project=benchmark benchmark/groupnormlinf.jl
```

## Cold-path harness

`l2_cold_paths.jl` is a differential check and a paired benchmark of the exceptional paths of `NormL2` and
`GroupNormL2` and of the shared kernels that serve them (issue #900). It has the design of `l2_norms.jl`
(roles X, Y and the A/A control XC compiled from source in one process, alternating order, inputs and output
buffers shared by the roles, `raw.csv` and `provenance.txt` in the format of
`summarize_norml1_infinite_weight.py`).

**Cases.** Every case is one operation, one element type and one input, with a name of the form
`family/operation/type/scenario/size[/layout]`.

| Family | Operations | Inputs |
|---|---|---|
| `GroupNormL2` | `prox`, `value_and_prox`, `value`, `prox_block`, `kkt_array`, `kkt_block`, and `*_matrix` for `Matrix` arguments | group sizes 2, 5, 32 and 1024 (and 2^16 in Float32), contiguous (range) and permuted (index-vector) groups, about 4096 entries |
| `NormL2` | `prox`, `value_and_prox`, `value` on arrays of 1024 and 65536 entries; `value_sparse`, `value_symmetric`, `value_block` | 400 stored entries of a 200 × 200 `SparseIterate` and `SymmetricSparseIterate`, a `BlockIterate` of 1024 entries |
| `SqrNormL2`, `norm_diff`, `LeastSquaresLoss`, `NormL1` | values (`NormL1` also `prox` and `value_and_prox`) | the paths of the shared kernels that the two norms use |

The scenarios are: overflowing norms (entries near `floatmax`, sums of squares that overflow with a finite
norm); underflowing norms (entries near `floatmin`, subnormal entries, sums of squares below `m floatmin/eps`);
thresholds `γλ0w` that overflow or underflow with finite results, also where only `γλ0` is out of range;
extreme finite products; values near `floatmax/2` and above it (Option C); values below `K floatmin` (#373);
long Float32 groups of 2^16 entries; zero and infinite levels, weights and steps; NaN and Inf at the first, middle
and last positions (NaN after Inf, NaN before Inf); zero and `-0.0` groups. BigFloat has a few representative
inputs of each operation; the BigFloat cases are checked but not timed.

**Validation of the inputs.** Before any role is compared, the data of each scenario are validated with
computations that use no package code, for the properties that the scenario lists in its `req` tuple: the ordinary
sum of squares in the element type overflows, or is zero or small although the group is not, the product `γλ0w` is
out of range in the element type (and in range in exact arithmetic where the scenario says so), the exact value lies
in a stated fraction of `floatmax`, the exact proximal point is finite in the element type (`:finite_result`, for the
Float32 and Float64 scenarios that list it). The assertions are scenario-specific: a scenario with `req = ()`
(`reg_lam0_ord` and `reg_lamInf_ord`, whose parameters decide the path) has none, the BigFloat scenarios are not
asserted to give finite results, and a non-finite output (NaN or Inf data, infinite parameters) is intentional in
the scenarios made for it. A failed assertion is an error that names the case. That the data have these
properties, or that an input was built, is not evidence that every operation takes its intended path; the exact
comparison of the trees below is the check.

**Reference check of values.** The scenarios marked `ref` (the value operations of, for example, `over_mid_val`,
`over_norm_val`, `val_near`, `val_over`, `val_tu` and their `NormL2`, `SqrNormL2`, `LeastSquaresLoss` and `NormL1`
counterparts, in Float32 and Float64) also require the value `a` of X to agree with a 256-bit BigFloat reference
`ref` that is computed from the inputs. With `e = T(ref)` (the reference rounded to nearest in T), `a` agrees when
`a === e`, or when both are finite and `|a - e| <= max(128 eps(T) |e|, 4 K nextfloat(zero(T)))`, computed in T,
where K is the count that the scenario gives with its reference (the number of groups for `GroupNormL2`, one or the number
of entries for the other families). So the bound is relative, 128 eps, only where
`|e| >= K nextfloat(zero(T)) / (32 eps(T))` (about `K 2^-131` in Float32 and `K 2^-1027` in Float64); below that it
is absolute, and a relative error above 128 eps is accepted, up to a total loss (`a = 0` for the smallest
subnormal `e` with `K = 1` agrees). The other operations are compared with the reference tree only.

**Differential.** The check is always exact (`===` per array element and per scalar; a BigFloat by precision,
sign and value). XC must equal X and Y must equal X for every case, so a one-bit difference anywhere stops the
run with the case name.

```sh
PB_FIRST=/path/to/reference-tree PB_SECOND="$PWD" PB_CHECK_ONLY=1 \
  julia --threads=1 --project=benchmark benchmark/l2_cold_paths.jl
```

`PB_FILTER` selects cases by a regular expression on their ids (for example `'^GroupNormL2/prox/Float32/'`);
`PB_STRIDE=k` keeps every k-th selected case (a smoke run of the timing code, for example with `PB_ROUNDS=1`);
`PB_INVENTORY=1` prints the number of cases of each family, operation and type. The harness builds one benchmark
object per role and case and times it with `gctrial = false` (no full collection before each timing, which would
otherwise dominate a run over thousands of short cases); the other settings are those of `l2_norms.jl`.

**Allocations.** The closures that the harness times are built by function barriers (`bindcall` and the like),
so that a call through the run-time module value does not allocate by itself. After warming, the check requires
`@allocated op() == 0` for role X in every timed case, and lists the cases where Y allocates more than X. The
exceptions, which are allocations of the package call and are listed by id when the check runs, are in the
`ALLOC_OK` table of the script with the reason. Cases whose level is a `BigFloat` with Float64 data are checked
but not timed (the arithmetic is BigFloat).

**The BigFloat exception (D-X).** One case, `GroupNormL2/value/BigFloat/dx_exponent_sum_beyond_Int64`, has
`λ0 = w = x = 2^(2^62-2)`, whose value overflows MPFR. The reference tree wraps the sum of three exponents and
returns `+0`; the shared kernel returns `+Inf`. The case is the only one allowed to differ. `PB_EXPECT_DIFF`
tells the check what to require of it:

| `PB_EXPECT_DIFF` | requirement |
|---|---|
| `auto` (default) | `same` when `PB_FIRST` and `PB_SECOND` are the same path, `forward` otherwise |
| `0` or `same` | X and Y are equal (use it when both roots hold the same source) |
| `1` or `forward` | X gives `+0` and Y gives `+Inf` (the reference tree is X) |
| `reversed` | X gives `+Inf` and Y gives `+0` (a run with the roles reversed) |

Run `l2_cold_paths.jl` with the roles reversed with `PB_EXPECT_DIFF=reversed` (or `0` when both trees hold the
same source); otherwise `auto` fails there.

## Timing campaign

The two harnesses time X, Y and XC in alternating rounds and write `raw.csv`; the check runs first and stops
at a mismatch. For the campaign, run each harness in a fresh process with one thread, on an otherwise idle
machine, pinned to one core (`taskset -c <core>` on Linux), forward and with the roles reversed, and
summarize each run on its own:

```sh
# forward: X = reference tree, Y = candidate
PB_FIRST=/path/to/reference-tree PB_SECOND="$PWD" PB_OUTPUT=/tmp/cold-fwd PB_ROUNDS=15 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/l2_cold_paths.jl
python3 benchmark/summarize_norml1_infinite_weight.py /tmp/cold-fwd

# reversed roles: X = candidate, Y = reference tree (the ratios are the inverse)
PB_FIRST="$PWD" PB_SECOND=/path/to/reference-tree PB_EXPECT_DIFF=reversed PB_OUTPUT=/tmp/cold-rev PB_ROUNDS=15 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/l2_cold_paths.jl
python3 benchmark/summarize_norml1_infinite_weight.py /tmp/cold-rev
```

The same commands with `benchmark/l2_norms.jl` time the ordinary cases (`PB_EXACT=1` applies to its check).
`--check-bounds=no` removes every bounds check, also those that the code does not elide with `@inbounds`;
run each harness again without the option (default bounds checking) and compare, because an access
that is checked in the shipped code (for example through an adapter whose reads are not marked `@inbounds`)
costs nothing in the first run and something in the second. Use a different `PB_OUTPUT` for each run:
the summarizer pools directories only when their `X:` and `Y:` lines of `provenance.txt` agree.
Judge a slowdown against the A/A column (`XC/X`) and repeat it with the roles reversed and in a fresh process
before calling it a regression; `PB_FILTER` restricts a repeat to the rows concerned.

## LinearFunction value harness

`linear_sparse_value.jl` times the `LinearFunction` value at a matrix `SparseIterate` (#810), which sums the stored
nonzero entries and reads the coefficients at them (a `Diagonal` coefficient array is summed over
the nonzero stored diagonal entries, `O(p)`), for Float32 and Float64 at 16 × 16,
128 × 128 and 1024 × 1024: nominal densities of 2 entries, 1%, 10%, 25% (shuffled), 50% and 100% (rounded to integer counts), a fully stored iterate with 50% and 90% of the
entries stored as zeros, sequential and shuffled activation order, an overflowing dot with finite coefficients
(the exceptional path), a `Diagonal` coefficient array, `value_and_gradient!` and `value_and_prox!` at a matrix
iterate, and controls for code that the change does not touch (the dense matrix value, the vector and the symmetric
iterate values, `gradient!` and `prox!` at a matrix iterate). It has the design of the harnesses above (roles X, Y
and the A/A control XC, alternating order, shared arrays, reversed-role and fresh-process repeats), and
`summarize_norml1_infinite_weight.py` summarizes its output. The roles can differ in the last bits of a value, because
the new value sums in another order than the base, so the roles' outputs are compared with each other with
tolerances (an empirical check for the random data of the cells, not an error bound); the exceptional cells are
compared with a BigFloat reference computed from the inputs, and a difference across the roles there is printed. The number of repetitions of a sample (`evals`) is a heuristic
from a nominal size of the cell (about its number of stored entries), and the duration of a sample is not calibrated. The iterates and their
backing arrays are built per role while the coefficient and output arrays are shared, so read timings through the A/A
spread, reversed-role and fresh-process repeats. With `PB_FILTER` the iterates of the other cells are not built (the
shared arrays of each size are). The usage is in the header of the script.

```sh
PB_FIRST=/path/to/tree-before PB_SECOND="$PWD" PB_CHECK_ONLY=1 \
  julia --threads=1 --project=benchmark benchmark/linear_sparse_value.jl
```

## Least-squares predictor harness

`leastsquares_predictor.jl` times the least-squares paths of `LeastSquaresLoss`, the `SqrNormL2` value with a
center, and the predictor and residual oracles `value_from_predictor`, `value_and_gradient_from_predictor!` and
`value_from_residual` (#915). It has the design of `l2_norms.jl` (roles X, Y and the A/A control XC compiled from
source in one process, alternating order, shared inputs and output buffers, `raw.csv` and `provenance.txt` in the
format of `summarize_norml1_infinite_weight.py`). The scratch `f.tmp` of the losses of the `LeastSquaresLoss` cells
and of the ordinary oracle cells is also shared, assigned after the constructor; the losses of the exceptional
oracle cells (`over_scaled`, `under_scaled`) keep the scratch of their constructor, which the value oracles timed
there do not use. The BLAS runs on one thread.

**Cases.** A case is one operation, one element type (Float32 or Float64) and one input, with an id
`family/operation/type/scenario/size`.

| Family | Operations | Inputs |
|---|---|---|
| `LeastSquaresLoss` | `value` at p = 6 and n = 64, 4096, 100000, raw and `Scaled(f, 1/n)`; `value_and_gradient` raw and scaled and `gradient` raw at (n, p) = (64, 6) and (2000, 500) | random data |
| `LeastSquaresLoss` | `value` at n = 64, p = 6: `under`, `over_val`, `exact_fit`, `prod_over` | squares that underflow to zero with a subnormal value; one square that overflows with a value below `floatmax`; an exact fit; a product `Xβ` that overflows (`Inf - Inf`) with `Xβ = 0` in exact arithmetic |
| `SqrNormL2` | `value` with a center of the data type and a level of the data type, and for Float32 data with a Float64 level (`center_lam64`), n = 64, 4096, 100000; `underflow` at n = 64 | random data; the underflow input of `under` |
| `LeastSquaresOracle` | the three oracles, raw (no scale), `scaled` (a scale 1/n of the data type) and for Float32 data `scaled_alpha64` (a Float64 scale 1/n), at n = 64, 4096, 100000 (the value oracles) and (n, p) = (64, 6) and (2000, 500) (value and gradient) | random data |
| `LeastSquaresOracle` | the two value oracles at n = 64 with `over_scaled` and `under_scaled` inputs (a sum of squares that overflows, or squares that underflow to zero, with a scale that brings the value into range) | constant residual vectors |
| `LeastSquaresOracle` | `value_from_predictor` and `value_from_residual`, raw, at n = 10⁶ and 10⁷ (p = 1) | random data; the check compares the oracle with a reference in extended precision and prints the error of each role in eps |

**Roles of the oracle cells.** The cells of the families `LeastSquaresLoss` and `SqrNormL2` (the first three rows of
the table) call the same methods in the three trees.
The oracle cells compare the oracles of the tree that has them with the loops that the script defines (value from
predictor `α (Σ (Yᵢ - zᵢ)² / 2)`, value and gradient `d = z - Y`, `g = Xᵀd`, `g .*= α`, `α (Σ dᵢ² / 2)`, value from residual
`α Σ rᵢ² / 2`; no `α` for the raw loss), which do not recover from overflow or underflow. A role whose tree has the
oracles calls them, and a role whose tree does not calls the loops. In a forward run (X = tree before, Y = tree
with the oracles) X and XC are the loops, compiled twice, and Y is the oracle; in a run with the roles reversed X
and XC are the oracle, compiled twice, and Y is the loop. XC/X is therefore the A/A control of these cells,
and Y/X compares different code.

**Check mode.** `PB_CHECK_ONLY=1` runs the check and the allocation measurement and exits.
- XC must equal X with `===`.
- On the ordinary cells Y must agree with X within 32 `eps(T)` (the largest difference per operation is printed;
  the bound is empirical for the random data of the cells, not an error bound). The long-vector cells (n = 10⁶ and
  10⁷) are left out of this comparison: the loops and the oracles sum differently there. Their oracle result must
  satisfy `|v - v*| <= (n + 8) eps(T) |v*|` against a reference `v*` in extended precision, and the error of each
  role is printed in eps.
- Each exceptional cell has a 256-bit BigFloat reference `v*` computed from its inputs. The result of the
  implementation from the tree with the new methods must be finite and satisfy `|v - v*| <= (n + 8) eps(T) |v*|`, plus `2 nextfloat(zero(T))` where
  `|v*| < floatmin(T)`. The result of the other implementation is printed, with whether it meets the bound.
- After warming, `@allocated op()` is measured for each role in each selected case behind a function barrier and printed.
  The check fails where the implementation from the tree with the new methods allocates more than the other, or allocates on an ordinary cell.

**Usage.**

```sh
PB_FIRST=/path/to/tree-before PB_SECOND="$PWD" PB_CHECK_ONLY=1 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/leastsquares_predictor.jl

# forward: X = tree before, Y = tree with the oracles
PB_FIRST=/path/to/tree-before PB_SECOND="$PWD" PB_OUTPUT=/tmp/ls-fwd PB_ROUNDS=15 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/leastsquares_predictor.jl
python3 benchmark/summarize_norml1_infinite_weight.py /tmp/ls-fwd

# reversed roles, summarized on its own
PB_FIRST="$PWD" PB_SECOND=/path/to/tree-before PB_OUTPUT=/tmp/ls-rev PB_ROUNDS=15 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/leastsquares_predictor.jl
python3 benchmark/summarize_norml1_infinite_weight.py /tmp/ls-rev
```

`PB_FILTER` selects cases by a regular expression on their ids (for example `'^LeastSquaresOracle/'`), `PB_ROUNDS`
sets the number of rounds, and `PB_PRINT=1` also prints `provenance.txt` and `raw.csv` to the standard output,
between the lines `BEGIN <file>` and `END <file>`, for a run whose output directory is not copied back.

**How to judge.** Judge each cell by its median paired ratio Y/X against the A/A ratio XC/X, in the forward and in
the reversed run (the ratios are the inverse in the reversed run). A cell above 1.05 in both orders that the A/A
spread does not explain is repeated in a fresh process with `PB_FILTER` before it is reported. The exceptional cells
are reported, not judged against 1.05. The oracle value cells at n = 64 take about 10 to 25 ns on the cluster nodes used, where a difference of 1 ns is
4 to 10 percent.

## Square-root least-squares harness

`sqrt_leastsquares.jl` times the value oracles `value_from_predictor` and `value_from_residual` of
`SqrtLeastSquaresLoss` (#960), raw and `Scaled(f, 1/√n)`, Float32 and Float64. It loads one tree (`PB_ROOT`, default
the repository) with `Base.include`, as the other harnesses do, and compares the oracles with a reference in the
same process; there are no roles X and Y.

| n | Reference | A/A control |
|---|---|---|
| 100, 10³, 10⁴, 10⁵ (one vectorizing loop) | a loop written in the script, `α √(Σ (zᵢ - Yᵢ)²)` or `α √(Σ rᵢ²)`, no `α` for the raw loss, no recovery from overflow or underflow | the same loop compiled again from the same source in a second module |
| 10⁶ (the pairwise reduction) | the oracle `value_from_predictor` or `value_from_residual` of `LeastSquaresLoss` of the same tree on the same data | the same least-squares oracle of a second compiled copy of the tree |

Each round times the three roles of a cell (reference, oracle, control) in an order that alternates between rounds,
and visits the cells in an order that alternates too; a timing is the median and the minimum over batches of calls.
The script prints, per cell, the median over the rounds of the ratio oracle/reference of the medians and of the
minima, the median over the rounds of the control's ratio of the medians, the range of the control's ratios and the
distance of their median from one, the warm allocations of each side (measured behind a function barrier, after the
check of the values), and a verdict: PASS where the median ratio is at most 1.25 (n = 100) or 1.10 (the other n);
otherwise INCONCLUSIVE where the median ratio minus one is at most the control's noise (the larger of its range and
its distance from one), FAIL otherwise. The value check is the comparison with the reference loop within 8 `eps(T)` for
n ≤ 10⁵. At n = 10⁶ the reference is a least-squares value, so the check compares `v_sqrt² / (2 v_LS)` with the factor
of the cell (1 for the raw loss) within 64 `eps(T)` relative, and the printed error is the relative error of that
quotient in `eps(T)`. A failed value check, or an oracle that allocates, is a failure. Run it again in a fresh process before reading a ratio near its limit:
at n = 100 a call takes 7 to 15 ns, where 1 ns is 7 to 14 percent, and the placement of code moves such a cell by
more than that.

```sh
PB_ROOT="$PWD" PB_OUTPUT=/tmp/sqrt-ls PB_ROUNDS=15 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/sqrt_leastsquares.jl
PB_ROOT="$PWD" PB_CHECK_ONLY=1 julia --threads=1 --check-bounds=no --project=benchmark benchmark/sqrt_leastsquares.jl
```

`PB_FILTER` selects cells by a regular expression on their ids `type/operation/variant/n` (for example
`'Float64/residual/.*/n100$'`), `PB_ROUNDS` sets the number of rounds, `PB_CHECK_ONLY=1` checks values and allocations
without timing, and `PB_OUTPUT` receives `raw.csv` and `provenance.txt`.

## Local `kkt_violation` harness

`kkt_local.jl` times the three- and four-argument `kkt_violation` between source trees compiled as modules of one process: `PB_BASE`
(required), `PB_HEAD` (default the repository), optionally earlier trees in `PB_HISTORY`, and a second compilation of `base` and of `head`
as the A/A controls (`basectl`, `headctl`). Group A times zero coordinates of the entrywise functions and `Scaled` of three of them; group B
all coordinates of points with a share of zeros and the scan of the inactive coordinates (the three-argument form is timed at array points)
for weighted and unweighted `NormL1` and `ElasticNet` and plain and centered `SqrNormL2`; group C `GroupNormL2`; group D the unweighted
`NormL1` against four reference kernels; group E one optimality check, in ns per inactive coordinate; group F a zero group of a
`BlockIterate` and a zero coordinate of a `SymmetricSparseIterate`. Before a case is timed the script compares each coordinate result (and
the scan lists) of the trees with `base`; if `base`, `basectl`, `head` and `headctl` differ it prints `VALUES DIFFER` and exits with a
failure status. `PB_CHECK_ONLY=1` runs the value and allocation checks without timing and ends with `CHECK PASS (<n> cases)` or `CHECK FAIL`.
`summarize_kkt_local.py` gives per row the median ns per call and the paired ratios `head/base` (`base/pre`, `head/pre` with a history tree
`pre`) with the controls and `head_vs_controls`. Compare a row with its two controls: rows of one loop shape and function move together, so
a difference of the size of the controls in one row is not evidence, and the same run in a second process with `PB_REVERSE=1` is the check.
The header of the script lists the settings and the output lines. It was written for issue #980.

```sh
PB_BASE=/path/to/tree-before PB_ROUNDS=15 julia --threads=1 --project=benchmark benchmark/kkt_local.jl > run.out
PB_BASE=/path/to/tree-before PB_REVERSE=1 julia --threads=1 --project=benchmark benchmark/kkt_local.jl > run-reversed.out
python3 benchmark/summarize_kkt_local.py summary.tsv run.out run-reversed.out
```

## Logistic-loss harness

Use a dedicated single-thread allocation for both source roles and the
independently compiled A/A control.

```sh
julia scripts/instantiate.jl benchmark
PB_BASELINE=/path/to/master PB_OUTPUT=/path/to/results PB_ROUNDS=7 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/logisticloss.jl
```

The logistic comparator is a standalone finite-data implementation with the
same design, coefficients, weights and offsets and one workspace. The
least-squares comparator loads the existing master source. Both comparisons
record paired medians, minima and warmed allocations in `raw.csv`, and the
source heads and Julia/BLAS settings in `provenance.txt`. The case ID names the
operation, precision, weighting/offset combination and shape. Compare the
candidate/reference ratios within each round to the control/reference ratios.
For an isolated compilation probe set `PB_FIRST_CALL=1` and select a single
case with `PB_FILTER` in each fresh process. The time in `first-call.csv`
includes the first invocation and its compilation, after loading source.

## Poisson-loss harness

`PoissonLoss` does not exist at the base commit `a38c0a8e6a6b7e8955dc784448542e4fd294c450`,
so this harness takes no baseline root and requires a candidate checkout that contains
`src/functions/poissonloss.jl`, and it loads the package source of that checkout
only. Use a dedicated single-thread allocation for the candidate and for the
independently compiled reference and A/A control.

```sh
julia scripts/instantiate.jl benchmark
PB_OUTPUT=/path/to/results PB_ROUNDS=7 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/poissonloss.jl
```

The reference is a standalone finite-data implementation with the same
design, coefficients, responses and products, one workspace and the same
scalar term as the candidate. The control is a second compilation of the
reference. The five operations are `f(β)`, `gradient!`, `value_and_gradient!`
and the two predictor oracles, at `(n, p)` = `(200, 20)`, `(2000, 100)`,
`(10000, 50)` and `(500, 2000)` with `Float64` data and ordinary predictors.
The case ID names the operation and the shape. `raw.csv` records paired
medians, minima and warmed allocations, and `provenance.txt` the source head and
the Julia/BLAS settings. Compare the candidate/reference ratios within each
round to the control/reference ratios. `PB_FILTER` selects cases by regular
expression on the case ID. For an isolated compilation probe set
`PB_FIRST_CALL=1` and select one case in each fresh process.

## SCAD

With separate clean worktrees for the candidate and `legacy/full`, run the
SCAD differential and time both implementations in the same exclusive
allocation. The candidate is also loaded under a second module for the A/A
comparison. The inputs cover the ordinary thresholding rules and the large-step
comparison away from switches where the two implementations differ.

```sh
julia scripts/instantiate.jl benchmark
PB_ARCHIVE=/path/to/archive PB_CANDIDATE="$PWD" PB_CHECK_ONLY=1 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/scad.jl
PB_ARCHIVE=/path/to/archive PB_CANDIDATE="$PWD" PB_OUTPUT=/tmp/scad-forward \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/scad.jl
PB_ARCHIVE=/path/to/archive PB_CANDIDATE="$PWD" PB_OUTPUT=/tmp/scad-reverse PB_REVERSE=1 \
  julia --threads=1 --check-bounds=no --project=benchmark benchmark/scad.jl
python3 benchmark/summarize_scad.py /tmp/scad-forward /tmp/scad-reverse
julia --threads=1 --project=. benchmark/scad_correctness.jl 200000
```

The timing harness warms every route, alternates measurement order and
records six paired rounds by default. It writes allocation counts and
paired times to `raw.csv`, individual samples to `samples.csv` and
revision, hardware and environment details to `provenance.txt`.

The `default_call` cases time `SCAD(0.5f0)`, `SCAD(0.5f0, 3.7f0)` and
`SCAD(0.5)` on matching Float32 or Float64 vectors. They measure `prox`
and `prox!` with the step omitted and with a typed step of one. The
`alloc_prox` cases include the allocation of the returned array. Use
`PB_FILTER='/default_call/'` for these cases alone.

Set `PB_PREOPT` to a clean checkout at
`2fd9cd416ab597ad05000cbad216a1db7b342308` to use the pre-optimization
candidate as the reference. Run and summarize each reference separately,
with independent forward and reverse output directories. The same A/A
control runs in both comparisons.

The independent correctness sweep uses 256-bit arithmetic for ordinary
inputs and 4096-bit arithmetic over the wider exponent range.

## NormL0 with diagonal = false

`norml0offdiag.jl` times `prox!`, `value_and_prox!` and the value of `NormL0(λ; diagonal = false)` against
the same operations of the all-entry `NormL0` on one symmetric matrix, for `p = 8, 32, 64` and
`Float32` and `Float64`, and the in-place forms of the first two (each copies the input into the
buffer first, for both functions). It checks that the two maps agree off the diagonal, alternates
the order of the paired samples and prints, for each operation, the fraction of off-diagonal entries
that the map keeps on the matrix of that run, rounded to two decimal places (`kept`), the median
time per call of each function over `PB_SAMPLES` samples, a positive count (the mean of the two
middle samples when the count is even), their ratio, the ratio of the
diagonal-free time to the out-of-place `prox!` of `NormL0`, and the bytes allocated by the
diagonal-free operation. It runs in the root project:

```sh
julia --threads=1 --project=. -e 'using Pkg; Pkg.instantiate(); include("benchmark/norml0offdiag.jl")'
```

Before the proximal map writes, `NormL0(λ; diagonal = false)` makes a finiteness pass over the entries of the
input; it does not compare the two triangles. One run on Julia 1.13.4 at commit `0d9ceb65b`, before the function
moved to the `diagonal` keyword of `NormL0`, gave these ratios to the same operation of `NormL0`. At that
commit the finiteness pass counted the non-finite entries column by column for every strided matrix; the
current code sums `x - x` over a `Matrix` in one pass (see the implementation notes of the
[numerical contract](../docs/src/spec/norml0offdiag.md)), so the table does not describe the current pass.
More than half of the off-diagonal entries of that run's matrices lay above the threshold. The allocation was zero bytes in each row of that
run. In this run the in-place ratios were below one at some sizes, and the in-place `prox!` of
`NormL0` was slower than its out-of-place `prox!`.

| Operation | `Float32`, `p = 8, 32, 64` | `Float64`, `p = 8, 32, 64` |
|:--|:--|:--|
| `prox!` | 4.7, 3.7, 3.7 | 3.4, 3.0, 1.7 |
| `value_and_prox!` | 3.6, 2.5, 2.1 | 2.8, 1.8, 1.3 |
| value | 5.7, 2.6, 1.9 | 3.8, 1.8, 1.4 |
| `prox!` in place | 1.3, 0.6, 0.5 | 1.2, 0.8, 0.7 |
| `value_and_prox!` in place | 1.5, 0.7, 0.5 | 1.3, 0.8, 0.7 |
