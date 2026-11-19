# Reproduce the ProximalBase 981 experiments

Fetch the experimental refs and use the frozen revisions in `pins.tsv`.
Run these commands from a fresh directory.

```bash
git init -q source
git -C source fetch https://github.com/mlakolar/ProximalBase.jl.git \
  'refs/heads/exp/981-*:refs/heads/exp/981-*'
git -C source worktree add --detach "$PWD/harness" exp/981-reproduce
data="$PWD/harness/experiment981/reproduction"
for spec in base:base A:kernel-reuse B:direct-face C:sort-projection; do
  git -C source worktree add --detach "$PWD/${spec%:*}" "exp/981-${spec#*:}"
done
cp "$data"/Manifest-v1.*.toml base/
cp "$data"/pins.tsv .
```

Instantiate the frozen project once with Julia 1.10.13 and once with Julia
1.13.4. In the following commands `JULIA` names the desired Julia executable.
Pass `-O0` on a separate accuracy run.

```bash
export JULIA=/absolute/path/julia
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$JULIA" --startup-file=no --project=base -e 'using Pkg; Pkg.instantiate()'
"$JULIA" --startup-file=no --project=base harness/experiment981/check_oracle.jl
for label in base A B C; do
  "$JULIA" --startup-file=no --project=base harness/experiment981/accuracy.jl \
    "$label" "$PWD/$label" "$label-default.tsv"
  "$JULIA" -O0 --startup-file=no --project=base harness/experiment981/accuracy.jl \
    "$label" "$PWD/$label" "$label-O0.tsv"
  "$JULIA" --startup-file=no --project=base harness/experiment981/reads.jl \
    "$label" "$PWD/$label" "reads-$label.tsv"
done
export N981_TARGET_NS=650000
for label in A B C base; do
  "$JULIA" --startup-file=no --project=base harness/experiment981/timing.jl \
    A "$PWD/A" "$label" "$PWD/$label" "timing-A-$label.tsv" 8
done
```

The experiment timings used a single exclusive Discovery node and a one-core
Slurm step. The batch script and its log record resource selection, CPU, and
affinity. To use that driver with these checkouts, copy the archived
`driver/` next to `base/` and submit `driver/batch.sh` with the experiment
directory as its argument. Its default Julia layout is
`$HOME/julia/julia-VERSION/bin/julia`.

For the independent review counterexamples run `driver/boundary.jl` with
each checkout as the Julia project. Copy both supplied manifests into that
checkout before using it directly as a Julia project.

```bash
cp "$data"/Manifest-v1.*.toml C/
"$JULIA" --startup-file=no --project=C "$data"/driver/boundary.jl
"$JULIA" -O0 --startup-file=no --project=C "$data"/review/mixed.jl
```

## Solver comparisons

For either follow-up, start in a fresh directory and set `data` to the
absolute reproduction directory in the harness checkout above. Set
`study` to `repair-solver` (baseline versus repaired sorting C) or
`pr-long-norm` (baseline versus pairwise norm).

```bash
git init -q source
git -C source fetch https://github.com/mlakolar/ProximalBase.jl.git \
  'refs/heads/exp/981-*:refs/heads/exp/981-*'
while read -r label sha; do
  git -C source worktree add --detach "$PWD/$label" "$sha"
done < "$data/$study/pins.tsv"
cp "$data/$study"/Manifest-v1.*.toml base/
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
"$JULIA" --startup-file=no --project=base -e 'using Pkg; Pkg.instantiate()'
mkdir -p results
if [ "$study" = repair-solver ]; then
  "$JULIA" --startup-file=no --project=base "$data/$study/paired.jl" "$PWD/PR" "$PWD/C" "$PWD/results" 12
  "$JULIA" --startup-file=no --project=base "$data/$study/profile.jl" "$PWD/PR" "$PWD/C" "$PWD/results" 15
else
  "$JULIA" --startup-file=no --project=base "$data/$study/paired.jl" "$PWD/PR" "$PWD/pairwise" "$PWD/results" 12
  "$JULIA" --startup-file=no --project=base "$data/$study/focused-bench.jl" "$PWD/PR" "$PWD/pairwise" "$PWD/results/scores.tsv" 12
fi
```

Use Julia 1.10.13 and 1.13.4 in separate output directories. For cluster
runs, the study's `batch.sh` and shell drivers record how the original
jobs paired the inputs and controlled threads. The historical comparator
is reachable at `exp/981-pr-comparator`.
