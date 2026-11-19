#!/usr/bin/env bash
set -euo pipefail
cd "$1"
export JULIA_DEPOT_PATH="$PWD/depot:$HOME/.julia:" JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 JULIA_NUM_THREADS=1
printf 'Job %s attempt %s on %s\n' "$SLURM_JOB_ID" "${SLURM_RESTART_COUNT:-0}" "$SLURMD_NODENAME"
scontrol show job --oneliner "$SLURM_JOB_ID"
squeue -h -w "$SLURMD_NODENAME" -o "%u %j %C"
lscpu | grep -E '^Model name|^Flags|^CPU\(s\):'
taskset -pc $$
for version in 1.10.13 1.13.4; do
    "$HOME/julia/julia-$version/bin/julia" --startup-file=no --threads=1 --project=base -e '
        using ProximalBase
        @assert kkt_violation(NormLinf(1.0), [2.0, 0.0], zeros(2)) ≈ 1.0
        println("smoke passed on ", VERSION)
    '
done
squeue -h -w "$SLURMD_NODENAME" -o "%u %j %C"
