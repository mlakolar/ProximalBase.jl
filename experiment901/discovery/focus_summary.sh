#!/bin/bash
# focus_summary.sh <downloaded focus run dir> <label> [cold|norms]  -> prints summarize901 output (label/base over all processes)
set -euo pipefail
H=$(cd "$(dirname "$0")/.." && pwd); D=$1; L=$2; K=${3:-cold}
OUT=$D/conv-$L-$K; rm -rf "$OUT"; mkdir -p "$OUT"; runs=()
for s in "$D"/${L}_${K}_*_r*; do
  [ -d "$s" ] || continue; b=$(basename "$s")
  if [[ $b == *_fwd_* ]]; then X=base; Y=$L; else X=$L; Y=base; fi
  python3 -I "$H/cold2screen.py" "$s" "$X" "$Y" "$OUT/$b"; runs+=("$OUT/$b")
done
python3 -I "$H/summarize901.py" --num "$L" --den base --top 40 "${runs[@]}"
