#!/bin/bash
set -euo pipefail
version="$1"; repetition="$2"
export PB_REPETITION="$repetition" PB_MODE=full PB_SAMPLES=40 PB_SECONDS=0.05 PB_PROFILE_SECONDS=0.1
export PB_HARNESS_SHA=5e92a5d6eac823b627afa2a3b06b5cb68f59eaf0
if (( repetition % 2 )); then variants=(before partial-v6); else variants=(partial-v6 before); fi
pair_status=0
for variant in "${variants[@]}"; do
  if [[ "$variant" == before ]]; then
    export PB_OUTPUT="$PWD/partial2-before" PB_SOURCE_SHA=0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a
  else
    export PB_OUTPUT="$PWD/partial2-after" PB_SOURCE_SHA=50ce5b52f6d3b58f07927bfa4500159c120a8d2f
  fi
  log="$PB_OUTPUT/julia-$version-rep$(printf '%02d' "$repetition")-worker.log"
  { date -u; hostname; grep Cpus_allowed_list /proc/self/status;
    "/home1/mkolar/juliaPkg/julia-$version/bin/julia" --startup-file=no --project="$PWD/env-$variant-$version" "$PWD/partial-v5/benchmark/changes/partial_svd.jl";
  } > "$log" 2>&1 || pair_status=1
done
exit "$pair_status"
