#!/bin/bash
set -euo pipefail
version="$1"; repetition="$2"
export PB_REPETITION="$repetition" PB_MODE=full PB_SAMPLES=40 PB_SECONDS=0.05 PB_PROFILE_SECONDS=0.1
export PB_HARNESS_SHA=f69cbe5733314c0cd80ab194c1a9fad377425dc1
export PB_MEASUREMENT_SCRIPT="$PWD/entropy-v1/benchmark/changes/entropy.jl"
if (( repetition % 2 )); then variants=(before entropy-v1); else variants=(entropy-v1 before); fi
pair_status=0
for variant in "${variants[@]}"; do
  export PB_EXPECTED_PACKAGE_ROOT="$PWD/$variant"
  if [[ "$variant" == before ]]; then
    export PB_OUTPUT="$PWD/entropy2-before" PB_SOURCE_SHA=0111d1bb4ca51b40c28d8ea5dab0f63be4d76e6a
    project="$PWD/env-before-$version"
  else
    export PB_OUTPUT="$PWD/entropy2-after" PB_SOURCE_SHA=f69cbe5733314c0cd80ab194c1a9fad377425dc1
    project="$PWD/env-entropy-fixed-$version"
  fi
  log="$PB_OUTPUT/julia-$version-rep$(printf '%02d' "$repetition")-worker.log"
  { date -u;hostname;grep Cpus_allowed_list /proc/self/status;
    "/home1/mkolar/juliaPkg/julia-$version/bin/julia" --startup-file=no --project="$project" -e 'using ProximalBase; actual=realpath(dirname(dirname(pathof(ProximalBase)))); expected=realpath(ENV["PB_EXPECTED_PACKAGE_ROOT"]); println("LOADED_PACKAGE_ROOT=",actual); actual==expected || error("wrong package source"); include(ENV["PB_MEASUREMENT_SCRIPT"])';
  } > "$log" 2>&1 || pair_status=1
done
exit "$pair_status"
