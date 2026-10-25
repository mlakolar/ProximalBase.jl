#!/bin/bash
set -euo pipefail
version="$1"; repetition="$2"
export PB_REPETITION="$repetition" PB_MODE=full PB_SAMPLES=40 PB_SECONDS=0.05 PB_PROFILE_SECONDS=0.1
export PB_HARNESS_SHA=1132145cf8ac216562176640aca99d8eed49add6
if (( repetition % 2 )); then variants=(before lq-v2); else variants=(lq-v2 before); fi
pair_status=0
for variant in "${variants[@]}"; do
  if [[ "$variant" == before ]]; then
    export PB_OUTPUT="$PWD/lq-before" PB_SOURCE_SHA=0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a
  else
    export PB_OUTPUT="$PWD/lq-after" PB_SOURCE_SHA=1132145cf8ac216562176640aca99d8eed49add6
  fi
  log="$PB_OUTPUT/julia-$version-rep$(printf '%02d' "$repetition")-worker.log"
  { date -u; hostname; grep Cpus_allowed_list /proc/self/status;
    "/home1/mkolar/juliaPkg/julia-$version/bin/julia" --startup-file=no --project="$PWD/env-$variant-$version" "$PWD/lq-v2/benchmark/changes/lq_half.jl";
  } > "$log" 2>&1 || pair_status=1
done
exit "$pair_status"
