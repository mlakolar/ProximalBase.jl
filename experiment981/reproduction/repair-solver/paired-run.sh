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
    target="$results/$version"
    if [[ -f "$target/timing-done" ]]; then continue; fi
    mkdir -p "$target"
    "$HOME/julia/julia-$version/bin/julia" --startup-file=no --threads=1 --project=base \
        driver/paired.jl "$root/PR" "$root/C" "$target" 12
    touch "$target/timing-done"
done
for version in 1.10.13 1.13.4; do
    target="$results/$version"
    if [[ -f "$target/done" ]]; then continue; fi
    "$HOME/julia/julia-$version/bin/julia" --startup-file=no --threads=1 --project=base \
        driver/profile.jl "$root/PR" "$root/C" "$target" 15
    touch "$target/done"
done
squeue -h -w "$SLURMD_NODENAME" -o '%u %j %C'
printf 'Finished at %s\n' "$(date -Is)"
