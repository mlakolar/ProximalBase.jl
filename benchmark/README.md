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
