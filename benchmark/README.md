# Minimal-core benchmark and independent references

The 102-case harness compares the retained storage/function hot paths to the
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

**Validation of the inputs.** Before any role is compared, each data set is validated with computations that use
no package code: the ordinary sum of squares in the element type overflows or underflows, the product `γλ0w`
is out of range in the element type and in range in exact arithmetic where the case says so, a value lies in the
intended fraction of `floatmax`, the exact BigFloat proximal point is finite. A failed assertion is an error that
names the case. Value cases with a BigFloat reference also require the value of X to lie within `128 eps(T)`
of it.

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
