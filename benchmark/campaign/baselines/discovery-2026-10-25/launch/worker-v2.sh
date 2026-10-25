#!/bin/bash
set -euo pipefail
slot="$1"; version="$2"; repetition="$3"
node=$(sed -n "${slot}p" "$PB_NODE_FILE")
[[ -n "$node" ]]
export PB_REPETITION="$repetition"
log="$PB_OUTPUT/julia-$version-rep$(printf '%02d' "$repetition")-worker.log"
srun --exclusive --exact -N1 -n1 -c1 --mem=4G --cpu-bind=cores --nodelist="$node" \
    bash -c 'date -u; hostname; cat /proc/loadavg; grep Cpus_allowed_list /proc/self/status; exec "$@"' \
    _ "/home1/mkolar/juliaPkg/julia-$version/bin/julia" --startup-file=no \
    --project="$SLURM_SUBMIT_DIR/env-v5-$version" "$PB_SOURCE/benchmark/campaign/run.jl" > "$log" 2>&1
