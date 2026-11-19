#!/usr/bin/env bash
set -euo pipefail

# Staged tree holds checked-out source snapshots named base and A/B/C, the
# pinned harness, and the versioned manifests used to load dependencies.
root=$(cd "$1" && pwd)
results=$(mkdir -p "$2" && cd "$2" && pwd)
versions=${N981_VERSIONS:-1.10.13 1.13.4}
targets=${N981_PAIRS:-A:A A:B A:C A:base}
reps=${N981_REPS:-8}
export JULIA_DEPOT_PATH="$root/depot:$HOME/.julia:" JULIA_PKG_OFFLINE=true JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export N981_TARGET_NS=${N981_TARGET_NS:-650000}

printf 'Job %s attempt %s on %s at %s\n' "${SLURM_JOB_ID:-local}" "${SLURM_RESTART_COUNT:-0}" "$(hostname -s)" "$(date -Is)"
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    scontrol show job --oneliner "$SLURM_JOB_ID"
    squeue -h -w "$SLURMD_NODENAME" -o '%u %j %C'
fi
uname -sr
lscpu
taskset -pc $$
cat "$root/pins.tsv"

for version in $versions; do
    julia="$HOME/julia/julia-$version/bin/julia"
    if [[ -n ${N981_JULIA_ROOT:-} ]]; then
        julia="$N981_JULIA_ROOT/julia-$version+0.x64.linux.gnu/bin/julia"
    fi
    "$julia" --version
    for pair in $targets; do
        left=${pair%:*}
        right=${pair#*:}
        file="$results/timing-$version-$left-$right.tsv"
        if [[ -s $file ]]; then
            continue
        fi
        echo "Running $version $left vs $right"
        "$julia" --startup-file=no --threads=1 --project="$root/base" "$root/harness/experiment981/timing.jl" \
            "$left" "$root/$left" "$right" "$root/$right" "$file.tmp" "$reps"
        mv "$file.tmp" "$file"
    done
done
if [[ -n ${SLURM_JOB_ID:-} ]]; then
    squeue -h -w "$SLURMD_NODENAME" -o '%u %j %C'
fi
printf 'Finished at %s\n' "$(date -Is)"
