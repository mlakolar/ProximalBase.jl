#!/bin/bash
set -euo pipefail
slot="$1"; version="$2"; repetition="$3"
node=$(sed -n "${slot}p" "$PB_NODE_FILE")
export PB_REPETITION="$repetition"
log="$PB_OUTPUT/julia-$version-rep$(printf '%02d' "$repetition")-worker.log"
srun --exclusive --exact -N1 -n1 -c1 --mem=4G --cpu-bind=cores --nodelist="$node" bash -c 'date -u; hostname;grep Cpus_allowed_list /proc/self/status;exec "$@"' _ "/home1/mkolar/juliaPkg/julia-$version/bin/julia" --startup-file=no --project="$PWD/env-before-$version" -e 'using ProximalBase; actual=realpath(dirname(dirname(pathof(ProximalBase))));println("LOADED_PACKAGE_ROOT=",actual);actual==realpath(ENV["PB_EXPECTED_PACKAGE_ROOT"]) || error("wrong package source");include(ENV["PB_MEASUREMENT_SCRIPT"])' > "$log" 2>&1
