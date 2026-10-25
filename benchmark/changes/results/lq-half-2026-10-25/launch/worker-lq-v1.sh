#!/bin/bash
set -euo pipefail
slot="$1"; version="$2"; repetition="$3"
node=$(sed -n "${slot}p" "$PB_NODE_FILE")
srun --exclusive --exact -N1 -n1 -c1 --mem=4G --cpu-bind=cores --nodelist="$node" bash "$PWD/pair-lq-v1.sh" "$version" "$repetition"
