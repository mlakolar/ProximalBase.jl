#!/usr/bin/env bash
set -euo pipefail
root=$1
results="$root/results/$SLURM_JOB_ID"
cd "$root"
export JULIA_DEPOT_PATH="$root/depot:$HOME/.julia:" JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
printf 'Job %s attempt %s on %s at %s\n' "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}" "$SLURMD_NODENAME" "$(date -Is)"
scontrol show job --oneliner "$SLURM_JOB_ID"
squeue -h -w "$SLURMD_NODENAME" -o '%u %j %C'
lscpu | grep -E 'Model name|^CPU\(s\)'
taskset -pc $$
cat pins.tsv
for version in 1.10.13 1.13.4; do
    for mode in default O0; do
        target="$results/$version-$mode"
        if [[ -f "$target/done" ]]; then continue; fi
        mkdir -p "$target"
        opts=()
        if [[ "$mode" == O0 ]]; then opts=(-O0); fi
        julia="$HOME/julia/julia-$version/bin/julia"
        run() { "$julia" --startup-file=no --threads=1 "${opts[@]}" --project=base "$@"; }
        if [[ ! -f "$target/solver-done" ]]; then
            run driver/paired.jl "$root/PR" "$root/pairwise" "$target" 12
            touch "$target/solver-done"
        fi
        run driver/focused-bench.jl "$root/PR" "$root/pairwise" "$target/scores.tsv" 12
        touch "$target/done"
    done
done
squeue -h -w "$SLURMD_NODENAME" -o '%u %j %C'
printf 'Finished at %s\n' "$(date -Is)"
